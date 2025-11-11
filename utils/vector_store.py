from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
import os
import faiss
import numpy as np
import pickle
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple, Union
from loguru import logger
from functools import lru_cache
import time
from collections import defaultdict

# ==== 常量 ====
MAX_COLLECTION_NAME_LEN = 128
BATCH_SIZE = 1000  # 批处理大小
EMBEDDING_CACHE_SIZE = 10000  # 嵌入缓存大小
AUTO_SAVE_THRESHOLD = 100000  # 自动保存阈值
MAX_WORKERS = 4  # 最大线程数


# ==== 自定义异常 ====
class CollectionNotFoundError(Exception):
    pass


class VectorStoreError(Exception):
    pass


class MyVectorStore:
    def __init__(
            self,
            local_path: str,
            embedding: Embeddings,
            allow_dangerous_deserialization: bool = True,
            batch_size: int = BATCH_SIZE,
            enable_cache: bool = True,
            auto_save_threshold: int = AUTO_SAVE_THRESHOLD
    ):
        self.local_path: str = local_path
        self.embedding: Embeddings = embedding
        self.allow_dangerous: bool = allow_dangerous_deserialization
        self.batch_size: int = batch_size
        self.enable_cache: bool = enable_cache
        self.auto_save_threshold: int = auto_save_threshold

        self._collections: Dict[str, FAISS] = {}
        self._collection_locks: Dict[str, threading.RLock] = defaultdict(threading.RLock)
        self._global_lock = threading.Lock()
        self._pending_saves: Dict[str, int] = defaultdict(int)  # 跟踪未保存的更改数量
        self._executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

        # 嵌入缓存
        if self.enable_cache:
            self._embedding_cache: Dict[str, List[float]] = {}
            self._cache_lock = threading.Lock()

        # 预计算嵌入维度
        self._embedding_dim: Optional[int] = None

    def __del__(self):
        """清理资源"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)

    def get_collection_lock(self, collection_name):
        # 双重检查加锁，保证同一时间只创建一个锁对象
        if collection_name not in self._collection_locks:
            with self._global_lock:
                if collection_name not in self._collection_locks:
                    self._collection_locks[collection_name] = threading.Lock()
        return self._collection_locks[collection_name]

    # ==== 缓存相关 ====
    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """获取缓存的嵌入"""
        if not self.enable_cache:
            return None
        with self._cache_lock:
            return self._embedding_cache.get(text)

    def _cache_embedding(self, text: str, embedding: List[float]) -> None:
        """缓存嵌入"""
        if not self.enable_cache:
            return
        with self._cache_lock:
            if len(self._embedding_cache) >= EMBEDDING_CACHE_SIZE:
                # 简单的LRU：删除第一个元素
                first_key = next(iter(self._embedding_cache))
                del self._embedding_cache[first_key]
            self._embedding_cache[text] = embedding

    def _get_embedding_dim(self) -> int:
        """获取嵌入维度，只计算一次"""
        if self._embedding_dim is None:
            test_embedding = self.embedding.embed_query('test')
            self._embedding_dim = len(test_embedding)
        return self._embedding_dim

    # ==== 名称规范化 ====
    @lru_cache(maxsize=1000)
    def _normalize_collection_name(self, collection_name: str) -> str:
        if len(collection_name) > MAX_COLLECTION_NAME_LEN:
            short_name = collection_name[:MAX_COLLECTION_NAME_LEN]
            logger.info(
                f"collection_name '{collection_name}' 超过最大长度{MAX_COLLECTION_NAME_LEN}，已自动截断为 '{short_name}'"
            )
            return short_name
        return collection_name

    # ==== 工具方法 ====
    def _get_collection_path(self, collection_name: str) -> str:
        return os.path.join(self.local_path, collection_name)

    def _ensure_collection_loaded(self, collection_name: str) -> FAISS:
        with self._collection_locks[collection_name]:
            if collection_name not in self._collections:
                self.read_collection(collection_name)
            return self._collections[collection_name]

    # ==== 批量嵌入计算 ====
    def _compute_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """批量计算嵌入，利用缓存"""
        if not self.enable_cache:
            return self.embedding.embed_documents(texts)

        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # 检查缓存
        for i, text in enumerate(texts):
            cached = self._get_cached_embedding(text)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        # 批量计算未缓存的嵌入
        if uncached_texts:
            new_embeddings = self.embedding.embed_documents(uncached_texts)
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embedding
                self._cache_embedding(texts[idx], embedding)

        return embeddings

    def _process_documents_in_batches(self, documents: List[Document]) -> Tuple[
        List[str], List[List[float]], List[Dict]]:
        """批量处理文档，返回文本、嵌入和元数据"""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        all_embeddings = []

        # 分批处理
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self._compute_embeddings_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)

        return texts, all_embeddings, metadatas

    # ==== 异步保存 ====
    def _async_save_collection(self, collection_name: str) -> None:
        """异步保存集合"""

        def save_task():
            try:
                collection_path = self._get_collection_path(collection_name)
                with self._collection_locks[collection_name]:
                    if collection_name in self._collections:
                        os.makedirs(collection_path, exist_ok=True)
                        self._collections[collection_name].save_local(collection_path)
                        self._pending_saves[collection_name] = 0
                        logger.debug(f"异步保存集合 '{collection_name}' 完成")
            except Exception as e:
                logger.error(f"异步保存集合 '{collection_name}' 失败: {str(e)}")

        self._executor.submit(save_task)

    def _should_auto_save(self, collection_name: str) -> bool:
        """检查是否应该自动保存"""
        return self._pending_saves[collection_name] >= self.auto_save_threshold

    # ==== 集合的加载、保存、删除 ====
    def read_collection(self, collection_name: str) -> None:
        collection_name = self._normalize_collection_name(collection_name)
        collection_path = self._get_collection_path(collection_name)

        if not os.path.exists(collection_path):
            raise FileNotFoundError(f"集合 '{collection_name}' 不存在。")

        with self._collection_locks[collection_name]:
            if collection_name in self._collections:
                raise VectorStoreError(f"集合 '{collection_name}' 已加载。")

            try:
                db = FAISS.load_local(
                    folder_path=collection_path,
                    embeddings=self.embedding,
                    allow_dangerous_deserialization=self.allow_dangerous
                )
                if not hasattr(db, 'index_to_docstore_id'):
                    raise VectorStoreError(f"集合 '{collection_name}' 格式不兼容")
                self._collections[collection_name] = db
                logger.debug(f"加载集合 '{collection_name}' 成功")
            except Exception as e:
                raise VectorStoreError(f"加载集合 '{collection_name}' 失败: {str(e)}") from e

    def _save_collection(self, collection_name: str) -> None:
        """同步保存集合"""
        collection_name = self._normalize_collection_name(collection_name)
        collection_path = self._get_collection_path(collection_name)

        try:
            with self._collection_locks[collection_name]:
                os.makedirs(collection_path, exist_ok=True)
                self._collections[collection_name].save_local(collection_path)
                self._pending_saves[collection_name] = 0
        except Exception as e:
            raise VectorStoreError(f"保存集合 '{collection_name}' 失败: {str(e)}") from e

    def force_save_all(self) -> None:
        """强制保存所有集合"""
        for collection_name in list(self._collections.keys()):
            if self._pending_saves[collection_name] > 0:
                self._save_collection(collection_name)

    def delete_collection(self, collection_name: str, remove_files: bool = False) -> None:
        collection_name = self._normalize_collection_name(collection_name)

        with self._collection_locks[collection_name]:
            if collection_name not in self._collections:
                raise CollectionNotFoundError(f"集合 '{collection_name}' 未加载")

            del self._collections[collection_name]
            self._pending_saves[collection_name] = 0

            if remove_files:
                collection_path = self._get_collection_path(collection_name)
                if os.path.exists(collection_path):
                    for f in os.listdir(collection_path):
                        try:
                            os.remove(os.path.join(collection_path, f))
                        except Exception:
                            pass
                    try:
                        os.rmdir(collection_path)
                    except Exception as e:
                        raise VectorStoreError(f"删除集合文件夹失败: {str(e)}") from e

    def list_collections(self) -> List[str]:
        if not os.path.exists(self.local_path):
            return []

        collections = []
        for name in os.listdir(self.local_path):
            dir_path = os.path.join(self.local_path, name)
            if os.path.isdir(dir_path):
                files = set(os.listdir(dir_path))
                if {'index.faiss', 'index.pkl'}.issubset(files):
                    collections.append(name)
        return collections

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        collection_name = self._normalize_collection_name(collection_name)
        db = self._ensure_collection_loaded(collection_name)

        with self._collection_locks[collection_name]:
            return {
                "document_count": len(db.index_to_docstore_id),
                "embedding_size": getattr(db.index, "d", None),
                "is_trained": getattr(db.index, "is_trained", None),
                "pending_saves": self._pending_saves[collection_name]
            }

    # ==== 优化的集合创建与添加 ====
    def create_collection(self, collection_name: str, documents: List[Document]) -> None:
        """创建一个集合"""
        collection_name = self._normalize_collection_name(collection_name)

        if not documents:
            raise ValueError("文档列表不能为空")

        if collection_name in self._collections or os.path.exists(self._get_collection_path(collection_name)):
            raise VectorStoreError(f"集合 '{collection_name}' 已存在")

        try:
            with self._collection_locks[collection_name]:
                # 批量处理文档
                texts, embeddings, metadatas = self._process_documents_in_batches(documents)

                # 创建优化的FAISS索引
                embedding_size = self._get_embedding_dim()

                # 对于大量数据，使用更高效的索引类型
                if len(documents) > 10000:
                    # 使用IVF索引提高搜索性能
                    quantizer = faiss.IndexFlatIP(embedding_size)
                    nlist = min(int(np.sqrt(len(documents))), 1000)
                    index = faiss.IndexIVFFlat(quantizer, embedding_size, nlist)

                    # 训练索引
                    embeddings_array = np.array(embeddings, dtype=np.float32)
                    index.train(embeddings_array)
                else:
                    index = faiss.IndexFlatIP(embedding_size)

                db = FAISS(
                    embedding_function=self.embedding,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={},
                )

                # 批量添加文档
                db.add_texts(texts, metadatas)

                self._collections[collection_name] = db
                self._pending_saves[collection_name] = len(documents)

                # 异步保存
                self._async_save_collection(collection_name)

                logger.info(f"创建集合 '{collection_name}' 成功，包含 {len(documents)} 个文档")

        except Exception as e:
            self._collections.pop(collection_name, None)
            raise VectorStoreError(f"创建集合失败: {str(e)}") from e

    def add_to_collection(self, collection_name: str, documents: List[Document]) -> None:
        """优化的添加文档到集合"""
        collection_name = self._normalize_collection_name(collection_name)

        if not documents:
            raise ValueError("文档列表不能为空")

        try:
            with self._collection_locks[collection_name]:
                if collection_name not in self._collections:
                    if os.path.exists(self._get_collection_path(collection_name)):
                        self.read_collection(collection_name)
                    else:
                        self.create_collection(collection_name, documents)
                        return

                # 批量处理文档
                texts, embeddings, metadatas = self._process_documents_in_batches(documents)

                # 批量添加到FAISS
                self._collections[collection_name].add_texts(texts, metadatas)

                # 更新待保存计数
                self._pending_saves[collection_name] += len(documents)

                # 检查是否需要自动保存
                if self._should_auto_save(collection_name):
                    self._async_save_collection(collection_name)

                logger.debug(f"向集合 '{collection_name}' 添加 {len(documents)} 个文档")

        except Exception as e:
            raise VectorStoreError(f"操作集合 '{collection_name}' 失败: {str(e)}") from e

    def add_embedding_to_collection(
            self,
            collection_name: str,
            documents: List[Document],
            embeddings: List[List[float]]
    ) -> None:
        """直接使用预计算的嵌入添加文档"""
        collection_name = self._normalize_collection_name(collection_name)

        if not documents:
            raise ValueError("文档列表不能为空")

        if len(documents) != len(embeddings):
            raise ValueError("文档数量与嵌入数量不匹配")

        try:
            with self.get_collection_lock(collection_name):
                if collection_name not in self._collections:
                    if os.path.exists(self._get_collection_path(collection_name)):
                        self.read_collection(collection_name)
                    else:
                        # 为预嵌入文档创建集合
                        embedding_size = len(embeddings[0])
                        index = faiss.IndexFlatIP(embedding_size)
                        db = FAISS(
                            embedding_function=self.embedding,
                            index=index,
                            docstore=InMemoryDocstore(),
                            index_to_docstore_id={},
                        )
                        self._collections[collection_name] = db

                # 批量添加预计算的嵌入
                texts = [doc.page_content for doc in documents]
                metadatas = [doc.metadata for doc in documents]

                # 缓存这些嵌入
                if self.enable_cache:
                    for text, embedding in zip(texts, embeddings):
                        self._cache_embedding(text, embedding)

                # 使用add_embeddings方法（如果支持）
                text_embedding_pairs = list(zip(texts, embeddings))
                self._collections[collection_name].add_embeddings(text_embedding_pairs, metadatas)

                self._pending_saves[collection_name] += len(documents)

                if self._should_auto_save(collection_name):
                    self._async_save_collection(collection_name)

        except Exception as e:
            raise VectorStoreError(f"操作集合 '{collection_name}' 失败: {str(e)}") from e

    # ==== 查询相关 ====
    def embedding_documents(self, documents: List[Document]) -> List[List[float]]:
        """批量计算文档嵌入"""
        if not documents:
            raise ValueError("文档列表不能为空")

        texts = [doc.page_content for doc in documents]
        return self._compute_embeddings_batch(texts)

    def query_collection_with_similarity_score(
            self,
            collection_name: str,
            query: str,
            k: int = 4,
            **kwargs
    ) -> List[Tuple[Document, float]]:
        collection_name = self._normalize_collection_name(collection_name)

        if not isinstance(k, int) or k <= 0:
            raise ValueError("k 必须是正整数")

        db = self._ensure_collection_loaded(collection_name)

        try:
            with self._collection_locks[collection_name]:
                max_k = min(k, len(db.index_to_docstore_id))
                return db.similarity_search_with_score(query, k=max_k, **kwargs)
        except Exception as e:
            raise VectorStoreError(f"查询执行失败: {str(e)}") from e

    def query_collection(
            self,
            collection_name: str,
            query: str,
            k: int = 4,
            **kwargs
    ) -> List[Document]:
        collection_name = self._normalize_collection_name(collection_name)
        return [doc for doc, _ in self.query_collection_with_similarity_score(collection_name, query, k, **kwargs)]

    def query_from_collections(
            self,
            collections: List[str],
            query: str,
            k: int = 4,
            **kwargs
    ) -> List[Document]:
        """并行查询多个集合"""
        docs_with_scores: List[Tuple[Document, float]] = []

        # 使用线程池并行查询
        futures = []
        for collection in collections:
            collection = self._normalize_collection_name(collection)
            future = self._executor.submit(
                self.query_collection_with_similarity_score,
                collection, query, k, **kwargs
            )
            futures.append(future)

        # 收集结果
        for future in as_completed(futures):
            try:
                docs_with_scores.extend(future.result())
            except Exception as e:
                logger.error(f"查询集合时出错: {str(e)}")

        # 排序并返回top-k
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in docs_with_scores[:k]]

    # ==== 性能监控 ====
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return {
            "loaded_collections": len(self._collections),
            "cache_size": len(self._embedding_cache) if self.enable_cache else 0,
            "pending_saves": dict(self._pending_saves),
            "embedding_dim": self._embedding_dim,
            "batch_size": self.batch_size,
            "auto_save_threshold": self.auto_save_threshold
        }

    def clear_cache(self) -> None:
        """清空嵌入缓存"""
        if self.enable_cache:
            with self._cache_lock:
                self._embedding_cache.clear()
                logger.info("嵌入缓存已清空")