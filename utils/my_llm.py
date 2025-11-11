# llm_client.py
from __future__ import annotations

import abc
import hashlib
import random
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterable, List, Optional, Protocol, Union, Callable

try:
    from loguru import logger
except Exception:  # 兜底到标准 logging
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("llm_client")

try:
    import tiktoken
except Exception:
    tiktoken = None  # 允许无 tiktoken 环境

from openai import OpenAI


# =========================
# 工具/协议/错误类型
# =========================

class CacheProtocol(Protocol):
    """最小缓存协议。"""

    def get(self, key: str) -> Optional[str]: ...

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None: ...


class LLMError(Exception):
    """通用 LLM 异常基类。"""
    pass


class OutOfBalanceError(LLMError):
    """余额不足类错误。"""
    pass


@dataclass(frozen=True)
class BackoffConfig:
    """重试退避策略配置。"""
    base_seconds: float = 1.0
    max_seconds: float = 30.0
    jitter: float = 0.1  # 相对抖动比例
    factor: float = 2.0  # 指数增长因子

    def sleep(self, attempt: int) -> None:
        delay = min(self.base_seconds * (self.factor ** attempt), self.max_seconds)
        # 加抖动
        jitter_val = delay * self.jitter * (random.random() * 2 - 1.0)
        time.sleep(max(0.0, delay + jitter_val))


@dataclass
class EndpointConfig:
    """单个端点（key/base_url/模型集）配置。"""
    api_key: str
    api_base_url: str
    candidate_models: List[str] = field(default_factory=lambda: ["gpt-4o-mini", "gpt-4o"])
    max_attempts: int = 5
    default_system_prompt: str = ""


@dataclass
class ClientConfig:
    """Client 层配置。"""
    temperature: float = 0.3
    top_p: Optional[float] = None
    stream: bool = False
    backoff: BackoffConfig = field(default_factory=BackoffConfig)
    cache: Optional[CacheProtocol] = None
    enable_cache_for_stream: bool = False  # 一般不建议对流式结果做缓存


# =========================
# Token 统计辅助
# =========================

def _token_len(text: str, model_hint: str = "gpt-4") -> int:
    """安全的 token 估算：优先 tiktoken，兜底粗略估算。"""
    if not text:
        return 0
    if tiktoken is None:
        # 粗略估算：≈4 chars/token
        return max(1, int(len(text) / 4))
    try:
        enc = tiktoken.encoding_for_model(model_hint)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# =========================
# 你的原始 _request_llm（保持语义）
# =========================

def _request_llm(
        client: OpenAI,
        prompt: str,
        model: str = 'gpt-4o',
        system: str = "",
        stream: bool = False,
        **kwargs: Any,
) -> Union[str, Iterable[str]]:  # 注意：流式时返回 Iterable[str]
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            stream=stream,
            **kwargs,
        )

        if not stream:
            if completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content
            return None
        else:
            def generate() -> Generator[str, None, None]:
                for chunk in completion:
                    # 有的块只有 finish_reason/role 等，没有 content，要判空
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        yield content

            return generate()
    except Exception:
        raise


# =========================
# 单端点 LLM 封装
# =========================

class SingleEndpointLLM:
    """
    单个端点的 LLM 调用器，负责：
    - 候选模型优先/回退
    - 重试 + 指数退避
    - 缓存
    - 令牌统计（流式/非流式）
    """

    def __init__(self, ep: EndpointConfig, cfg: ClientConfig):
        self._ep = ep
        self._cfg = cfg
        self._client = OpenAI(api_key=ep.api_key, base_url=ep.api_base_url)

    @staticmethod
    def _hash_key(*parts: str) -> str:
        h = hashlib.md5("||".join(parts).encode()).hexdigest()
        return "LLM_" + h[-15:]

    def call_llm(
            self,
            prompt: str,
            *,
            system: Optional[str] = None,
            stream: Optional[bool] = None,
            token_counter: Optional[Dict[str, Optional[int]]] = None,
            post_process: Optional[Callable[[str], str]] = None,
            request_overrides: Optional[Dict[str, Any]] = None,
            use_cache: bool = True,
            cache_token: Optional[str] = None,
            **kwargs
    ) -> Union[str, Generator[str, None, None]]:
        """
        Args:
            prompt: 用户输入
            system: SYSTEM 提示，默认取端点默认值
            stream: 是否流式；默认取全局配置
            token_counter: 若提供，将更新 {"input_token": int, "output_token": int}
            post_process: 非流式后处理函数
            request_overrides: 透传到 OpenAI 请求的额外参数（如 presence_penalty 等）

        Returns:
            - 非流式：str
            - 流式：生成器，逐步产出 str 片段
        """
        sys_msg = self._ep.default_system_prompt if system is None else system
        do_stream = self._cfg.stream if stream is None else stream
        post_process = post_process or (lambda s: s.strip() if isinstance(s, str) else s)
        ro = request_overrides or {}

        # --- token 统计：输入侧（粗略估算，不含 role/格式 token） ---
        if token_counter is not None:
            token_counter.clear()
            token_counter.update({
                "input_token": _token_len((sys_msg or "") + prompt, self._ep.candidate_models[0]),
                "output_token": None
            })

        cache_key = f"{cache_token}:{self._hash_key(str(self._ep.candidate_models), sys_msg or '', prompt)}"
        allow_cache = (self._cfg.cache is not None) and (not do_stream or self._cfg.enable_cache_for_stream) and use_cache

        # 1) 缓存命中（仅非流式通常使用）
        if allow_cache and not do_stream:
            cached = self._cfg.cache.get(cache_key)  # type: ignore[arg-type]
            if cached:
                if token_counter is not None:
                    token_counter["output_token"] = _token_len(cached, self._ep.candidate_models[0])
                return post_process(cached, **kwargs)

        # 2) 重试 + 候选模型回退
        last_err: Optional[Exception] = None
        for attempt in range(self._ep.max_attempts):
            model = self._select_model(attempt)

            try:
                if do_stream:
                    # --- 流式 ---
                    return self._stream_call(model, prompt, sys_msg, token_counter, ro)
                else:
                    # --- 非流式 ---
                    res = _request_llm(
                        client=self._client,
                        model=model,
                        prompt=prompt,
                        system=sys_msg or "",
                        stream=False,
                        temperature=self._cfg.temperature,
                        top_p=self._cfg.top_p,
                        **ro,
                    )
                    if not isinstance(res, str):
                        res = "" if res is None else str(res)

                    # 缓存
                    if allow_cache and res:
                        self._cfg.cache.set(cache_key, res)  # type: ignore[arg-type]

                    # 输出 token 统计
                    if token_counter is not None:
                        token_counter["output_token"] = _token_len(res, model)

                    return post_process(res, **kwargs)

            except Exception as e:
                last_err = e
                self._handle_error(attempt, e, model)
                if attempt < self._ep.max_attempts - 1:
                    self._cfg.backoff.sleep(attempt)
                    continue
                break

        # 3) 所有尝试均失败
        assert last_err is not None
        raise last_err

    def _stream_call(
            self,
            model: str,
            prompt: str,
            sys_msg: str,
            token_counter: Optional[Dict[str, Optional[int]]],
            request_overrides: Dict[str, Any],
    ) -> Generator[str, None, None]:
        gen = _request_llm(
            client=self._client,
            model=model,
            prompt=prompt,
            system=sys_msg or "",
            stream=True,
            temperature=self._cfg.temperature,
            top_p=self._cfg.top_p,
            **request_overrides,
        )

        output_accum: List[str] = []

        def _iter() -> Generator[str, None, None]:
            try:
                for piece in gen:  # 现在 gen 产出的是 str
                    if piece:
                        output_accum.append(piece)
                        yield piece
            finally:
                if token_counter is not None:
                    full = "".join(output_accum)
                    token_counter["output_token"] = _token_len(full, model)

        return _iter()

    def _select_model(self, attempt: int) -> str:
        """回退策略：前 N-1 次用第一个，最后一次用最后一个。也可自定义策略。"""
        if attempt == self._ep.max_attempts - 1 and len(self._ep.candidate_models) > 1:
            return self._ep.candidate_models[-1]
        return self._ep.candidate_models[0]

    def _handle_error(self, attempt: int, err: Exception, model: str) -> None:
        """统一错误处理与判别（可按需扩展错误码/文案）。"""
        msg = str(err)
        tb = traceback.format_exc()
        # 余额相关关键字
        if "30001" in msg or "余额不足" in msg or "insufficient_quota" in msg:
            logger.error(f"[OutOfBalance] model={model}, attempt={attempt}: {msg}")
            raise OutOfBalanceError(msg)

        # 常见限流/服务不可用
        if "429" in msg or "Rate limit" in msg:
            logger.warning(f"[RateLimit] model={model}, attempt={attempt}: {msg}")
            time.sleep(5)
        elif "503" in msg or "Service Unavailable" in msg:
            logger.warning(f"[ServiceUnavailable] model={model}, attempt={attempt}: {msg}")
        else:
            logger.warning(f"[LLMError] model={model}, attempt={attempt}: {msg}\n{tb}")


# =========================
# 多端点聚合 Client
# =========================

class LLMClient:
    def __init__(self, endpoints: List[Dict[str, Any]], client_cfg: Optional[ClientConfig] = None):
        self._cfg = client_cfg or ClientConfig()
        self._endpoint_cfgs = endpoints
        self._engines: Dict[int, SingleEndpointLLM] = {}  # 延迟初始化
        logger.info(f"Configured endpoints: {len(endpoints)}")

    def _get_engine(self, idx: int) -> SingleEndpointLLM:
        if idx not in self._engines:
            self._engines[idx] = SingleEndpointLLM(EndpointConfig(**self._endpoint_cfgs[idx]), self._cfg)
        return self._engines[idx]

    def call_llm(self, prompt: str, **kwargs: Any):
        if not self._endpoint_cfgs:
            raise LLMError("没有可用的 LLM 端点")

        while self._endpoint_cfgs:
            idx = random.randint(0, len(self._endpoint_cfgs) - 1)
            engine = self._get_engine(idx)   # 在这里才真正初始化
            try:
                return engine.call_llm(prompt, **kwargs)
            except OutOfBalanceError:
                logger.warning(f"移除余额不足端点，剩余: {len(self._endpoint_cfgs) - 1}")
                self._endpoint_cfgs.pop(idx)
                self._engines.pop(idx, None)
                if not self._endpoint_cfgs:
                    raise OutOfBalanceError("所有端点余额不足")
            except Exception:
                raise

