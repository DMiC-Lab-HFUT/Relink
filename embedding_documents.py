"""
多线程版本的文档向量化脚本
====================================================
功能：
    - 读取数据集文件（JSON 格式），并将其中的文档内容写入向量数据库
    - 支持多线程并发写入，提高处理效率
    - 关键步骤均配有中文注释，方便阅读与维护
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import List

from tqdm import tqdm
from langchain_core.documents import Document

from utils.embeddings import CustomEmbedding
from utils.vector_store import MyVectorStore
import configs
from utils.neo4j_operator import sanitize_label
import tiktoken


def split_text(text: str,
               overlap: int = 16,
               max_chunk_size: int = 128,
               min_chunk_size: int = 100,
               padding: str = " ...",
               model_name: str = 'gpt-3.5-turbo') -> List[str]:
    """
    将长文本分割成较小的块，并添加重叠和填充。

    Args:
        text (str): 输入文本。
        overlap (int): 块之间的重叠大小（token数）。
        max_chunk_size (int): 单个块的最大token数。
        min_chunk_size (int): 单个块的最小token数，用于合并小块。
        padding (str): 用于块之间连接的填充字符串。
        model_name (str): 用于编码文本的模型名称，决定tokenization方式。

    Returns:
        List[str]: 分割并填充后的文本块列表。
    """
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)

    step_size = max_chunk_size - overlap
    pos = 0
    chunks = []

    while pos < len(tokens):
        end_pos = pos + max_chunk_size

        if end_pos >= len(tokens):
            chunk = tokens[pos:len(tokens)]
            if len(chunk) < min_chunk_size and chunks:
                chunks[-1].extend(chunk)
            else:
                chunks.append(chunk)
            break
        else:
            chunk = tokens[pos:end_pos]
            chunks.append(chunk)
            pos += step_size

    texts = [encoding.decode(chunk) for chunk in chunks]

    padded_texts = []
    num_chunks = len(texts)

    if num_chunks <= 1:
        return texts

    for i, chunk_text in enumerate(texts):
        if i == 0:
            padded_chunk = chunk_text + padding
        elif i == num_chunks - 1:
            padded_chunk = padding + chunk_text
        else:
            padded_chunk = padding + chunk_text + padding
        padded_texts.append(padded_chunk)
    return padded_texts


def process_sample(sample: dict, vector_store: MyVectorStore, lock: Lock) -> None:
    """处理单个样本，并写入向量数据库

    Args:
        sample (dict): 单条样本数据，包含 question 与 context
        vector_store (MyVectorStore): 向量库实例
        lock (Lock): 线程锁，用于保护写操作
    """
    query = sample["question"]

    # 遍历每篇文章，将句子切分为文档
    for article in sample["context"]:
        documents = []
        title = article[0].strip()

        chunks = []
        for para in article[1]:
             chunks.extend(split_text(para))

        for chunk in chunks:
            # 过滤太短的句子，减少噪声
            if len(chunk.strip()) > 15:
                # 构造 page_content，格式：标题 | 句子
                page_content = f"{title} | {chunk.strip()}"
                meta = {
                    "title": title,
                    "sentence": chunk.strip(),
                }
                documents.append(Document(page_content=page_content, metadata=meta))

        if not documents:
            continue

        try:
            # 生成向量嵌入
            embeddings = vector_store.embedding_documents(documents)

            # 加锁写入，确保线程安全
            with lock:
                vector_store.add_embedding_to_collection(sanitize_label(query), documents, embeddings)
                vector_store.add_embedding_to_collection(sanitize_label(title), documents, embeddings)
                vector_store.add_embedding_to_collection(configs.DATASET_CONFIG["domain"], documents, embeddings)
        except Exception as e:
            raise e



def documents_to_vector_store(max_workers: int | None = None) -> None:
    """主函数：加载数据集并写入向量数据库（多线程）

    Args:
        max_workers (int | None): 线程池大小，默认使用 os.cpu_count()
    """

    # 初始化 Embedding 对象
    embeddings = CustomEmbedding(
        api_key=configs.EMBEDDING_CONFIG["api_key"],
        base_url=configs.EMBEDDING_CONFIG["base_url"],
        model_name=configs.EMBEDDING_CONFIG["model_name"],
    )

    # 创建向量库存储路径
    os.makedirs(configs.DATASET_CONFIG["doc_vector_store_path"], exist_ok=True)

    # 初始化向量库实例
    vector_store = MyVectorStore(configs.DATASET_CONFIG["doc_vector_store_path"], embeddings)

    # 加载数据集文件
    try:
        with open(configs.DATASET_CONFIG["dataset_file"], encoding="utf-8") as f:
            samples = json.load(f)
    except Exception as e:
        print(f"Failed to load {configs.DATASET_CONFIG['document_file']}: {e}")
        return

    # 线程锁，保护写操作
    lock = Lock()

    # 若未指定线程数，则使用 CPU 核心数
    if max_workers is None:
        max_workers = os.cpu_count() or 2

    # 创建线程池并执行任务
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(lambda s: process_sample(s, vector_store, lock), samples),
                total=len(samples),
                desc="Processing",
            )
        )
    vector_store.force_save_all()
    print(vector_store.get_performance_stats())


# ------------------------- 脚本入口 -------------------------
if __name__ == "__main__":
    documents_to_vector_store(max_workers=20)
