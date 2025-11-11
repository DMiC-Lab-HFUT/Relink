import time
from typing import List, Any, Callable
from langchain.embeddings.base import Embeddings
from openai import OpenAI
import numpy as np
from loguru import logger


def _batch_embed_with_retry(
        embedding_func: Callable[[List[str]], List[List[float]]],
        texts: List[str],
        batch_size: int = 100,
        max_retries: int = 5,
        retry_interval: int = 4
) -> List[List[float]]:
    """Process text embeddings in batches with retry mechanism.

    Args:
        embedding_func: Function to process a single batch of texts
        texts: List of texts to embed
        batch_size: Number of texts per batch
        max_retries: Maximum number of retry attempts
        retry_interval: Seconds to wait between retries

    Returns:
        List of embeddings for all input texts

    Raises:
        RuntimeError: If all retry attempts fail
    """
    for attempt in range(max_retries + 1):
        try:
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = [text or "null" for text in texts[i:i + batch_size]]
                embeddings.extend(embedding_func(batch))
            return embeddings
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                time.sleep(retry_interval)
            else:
                raise e
    return []  # 实际不可达，因最后会抛出异常


class CustomEmbedding(Embeddings):
    """Doubao API implementation for text embeddings with normalization.

    Attributes:
        model_name: Identifier for the embedding model
        client: Configured OpenAI client instance
        normalize: Whether to normalize the embeddings (default: True)
    """

    def __init__(
            self,
            model_name: str = "doubao-embedding-text-240715",
            api_key: str = "",
            base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
            dimensions: int = 1536,
            normalize: bool = True  # 新增参数：是否归一化
    ) -> None:
        """Initialize with API credentials and normalization option.

        Args:
            model_name: Identifier for the embedding model
            api_key: API key for authentication
            base_url: Base API endpoint URL
            normalize: Whether to normalize embeddings (default: True)
            **kwargs: Additional arguments for base class

        Raises:
            ValueError: If missing required API credentials
        """
        super().__init__()
        if not api_key or not base_url:
            raise ValueError("API key and base URL are required")
        self.dimensions = dimensions
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.normalize = normalize  # 新增归一化标志

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents with normalization.

        Args:
            texts: List of texts to embed

        Returns:
            List of normalized embedding vectors for each text
        """

        def _process_batch(batch: List[str]) -> List[List[float]]:
            """Internal batch processing with normalization"""
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
                dimensions=self.dimensions
            )
            embeddings = [item.embedding for item in response.data]

            if not self.normalize:
                return embeddings  # 不归一化则直接返回

            # 转换为numpy数组进行批量归一化
            embeddings_array = np.array(embeddings)
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)

            # 避免除以零（极少数情况）
            norms[norms == 0] = 1.0  # 或者处理为其他默认值

            normalized_embeddings = embeddings_array / norms
            return normalized_embeddings.tolist()

        return _batch_embed_with_retry(
            embedding_func=_process_batch,
            texts=texts,
            batch_size=100,
            max_retries=5,
            retry_interval=4
        )

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query with normalization.

        Args:
            text: Input text to embed

        Returns:
            Normalized embedding vector for the input text
        """
        return self.embed_documents([text])[0]



