import redis
from utils.my_llm import LLMClient, ClientConfig
from configs import (
    LLM_CONFIG,
    REDIS_CONFIG,
)
from core.llm_functions import *
from configs import logger


class NaiveGenerator:
    def __init__(self):
        self.init_config()

    def init_config(self):

        self.redis_client = redis.Redis(**REDIS_CONFIG)
        self.llm = LLMClient(endpoints=LLM_CONFIG['endpoints'], client_cfg=ClientConfig(cache=self.redis_client))

    def generate_answer(self, text, query):
        logger.debug(f"Generating the answer...")
        ans = reasoning_RAG(self.llm, text, query)
        logger.info(f"Answer: {ans}")
        return ans
    def generate_answer_without_context(self, query):
        logger.debug(f"Generating the answer...")
        ans = reasoning_LLM_only(self.llm, query)
        logger.info(f"Answer: {ans}")
        return ans

class MyGenerator:
    def __init__(self):
        self.init_config()

    def init_config(self):

        self.redis_client = redis.Redis(**REDIS_CONFIG)
        self.llm = LLMClient(endpoints=LLM_CONFIG['endpoints'], client_cfg=ClientConfig(cache=self.redis_client))

    def _format_context(self, paths):
        triples = [
            f"<{triple['begin']['mention']} | {triple['r'].replace('_', ' ')} | {triple['end']['mention']}>"
            for path in paths for triple in path.get('relations', [])
        ]
        format_triples = list(dict.fromkeys(triples))


        related_sents = [
            sent for path in paths for sent in path.get('context_sentences', [])
        ]
        related_sents = list(dict.fromkeys(related_sents))

        parts = []
        if format_triples:
            parts.append("Triples:\n" + "\n".join(format_triples))
        if related_sents:
            parts.append("Related Text:\n" + "\n".join(related_sents))
        context_info = "\n\n".join(parts)
        return context_info

    def generate_answer(self, paths, query):
        context_info = self._format_context(paths)
        logger.debug(f"Generating the answer...")
        ans = reasoning(self.llm, context_info, query)
        logger.info(f"Answer: {ans}")
        return ans
