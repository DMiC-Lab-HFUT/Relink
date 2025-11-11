import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
from utils.my_llm import LLMClient
from tqdm import tqdm
import redis
import re
import tiktoken
import copy
import os
from utils.neo4j_operator import deduplicates_list

import configs
from utils.my_llm import ClientConfig

# --- 配置和初始化 ---

# 定义文件路径常量
DOCUMENT_FILE = f"{configs.DATASET_CONFIG['dataset_path']}/dataset/documents.json"
KG_OUTPUT_FILE = f"{configs.DATASET_CONFIG['dataset_path']}/output/KG_extraction_results.json"
TRIPLE_OUTPUT_FILE = f"{configs.DATASET_CONFIG['dataset_path']}/output/extracted_triples.json"

# 确保输出目录存在
os.makedirs(f"{configs.DATASET_CONFIG['dataset_path']}/output", exist_ok=True)

# 初始化tiktoken编码器
# 注意：这里使用了 'gpt-4'，但split_text函数内部使用了 'gpt-3.5-turbo'，请根据实际需求统一
_global_encoder = tiktoken.encoding_for_model('gpt-4')


# --- 文档处理函数 ---
def split_text(text: str,
               overlap: int = 16,
               max_chunk_size: int = 128,
               min_chunk_size: int = 100,
               padding: str = " ...",
               model_name: str = 'gpt-3.5-turbo') -> List[str]:
    """
    Split long text into smaller chunks with overlapping and padding.

    Args:
        text (str): Input text.
        overlap (int): Overlap size between chunks in number of tokens.
        max_chunk_size (int): Maximum token count for a single chunk.
        min_chunk_size (int): Minimum token count for a single chunk, used to merge small chunks.
        padding (str): Padding string used to connect chunks.
        model_name (str): Model name for encoding the text, determines tokenization method.

    Returns:
        List[str]: List of split and padded text chunks.
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


def load_and_chunk_documents(file_path: str,
                             document_limit: int,
                             chunking_params: Dict) -> List[Dict]:
    """
    Load documents from the specified file and chunk the text of each document.

    Args:
        file_path (str): Path to the JSON file containing the documents.
        document_limit (int): Maximum number of documents to load.
        chunking_params (Dict): Dictionary containing chunking parameters, such as 'overlap', 'max_chunk_size', 'min_chunk_size', and 'padding'.

    Returns:
        List[Dict]: A list of document dictionaries with chunked text.
    """
    with open(file_path, encoding="utf-8") as f:
        documents = json.load(f)
    documents = deduplicates_list(documents)
    chunk_list = []
    for doc in documents[:document_limit]:
        # 使用传入的分块参数
        chunks = split_text(doc['text'], **chunking_params)
        for chunk in chunks:
            d = copy.deepcopy(doc)
            d['text'] = chunk
            chunk_list.append(d)
    return chunk_list


def process_single_document_for_kg(document: Dict, llm_client: Any) -> Dict:
    """
    Process a single document to extract knowledge graph triples.

    Args:
        document (Dict): A document dictionary containing 'title' and 'text'.
        llm_client (Any): An instance of an LLM client.

    Returns:
        Dict: A dictionary containing extraction results or error messages.
    """
    try:
        from core.llm_functions import extract_triples_from_doc
        document_text = f"{document['title']}\n{document['text']}"
        entities, relations = extract_triples_from_doc(llm_client, document_text)
        if hasattr(llm_client, 'last_response_tokens'):
            configs.TOKEN_COUNT += llm_client.last_response_tokens

        return {
            "document_id": document.get("id"),
            "title": document["title"],
            "entities": entities,
            "relations": relations,
            'meta': document.get("meta"),
            "status": "success"
        }
    except Exception as e:
        return {
            "document_id": document.get("id"),
            "title": document["title"],
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "failed"
        }


def process_documents_batch(documents: List[Dict],
                            llm_client: Any,
                            use_multithreading: bool = True,
                            max_workers: int = 4) -> List[Dict]:
    """
    Process documents in batch, either in parallel or sequentially.

    Args:
        documents (List[Dict]): List of documents to be processed.
        llm_client (Any): Instance of the LLM client.
        use_multithreading (bool): Whether to use multithreading.
        max_workers (int): Maximum number of worker threads when using multithreading.

    Returns:
        List[Dict]: List of processing results, maintaining the same order as the input documents.
    """
    results = [None] * len(documents)

    if use_multithreading:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(process_single_document_for_kg, doc, llm_client): idx
                               for idx, doc in enumerate(documents)}
            with tqdm(total=len(documents), desc="Processing Documents") as pbar:
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    results[idx] = future.result()
                    pbar.update(1)
    else:
        for idx, doc in enumerate(tqdm(documents, desc="Processing Documents")):
            results[idx] = process_single_document_for_kg(doc, llm_client)

    return results


def extract_triples_from_kg_results(kg_data: List[Dict]) -> List[Dict]:
    """
    Extract structured triples from the knowledge graph extraction results.

    Args:
        kg_data (List[Dict]): List of knowledge graph extraction results.

    Returns:
        List[Dict]: List of extracted triples.
    """
    knowledge_triples = []

    # 过滤掉失败的或不完整的结果
    filtered_kg_data = [entry for entry in kg_data if entry.get('status') == 'success'
                        and entry.get("entities") is not None
                        and entry.get("relations") is not None]

    for entry in filtered_kg_data:
        entities = entry["entities"]
        relationships = entry["relations"]
        entity_map = {e["id"]: e for e in entities}

        for rel in relationships:
            try:
                # 检查 source 和 target 是否在 entity_map 中
                if rel["source"] in entity_map and rel["target"] in entity_map:
                    triple = {
                        "subject": entity_map[rel["source"]],
                        "object": entity_map[rel["target"]],
                        "relation": rel["relation"],
                        'evidence': rel.get("evidence"),  # 使用 .get() 防止 Key Error
                        'meta': entry.get('meta')
                    }
                    knowledge_triples.append(triple)
                else:
                    print(
                        f"`Warning: The relationship {rel.get('relation', 'unknown')} contains missing entities (source: {rel.get('source')}, target: {rel.get('target')}), skipping this relationship.`")
            except Exception as e:  # 捕获其他潜在错误
                print(f"An error occurred while processing the relationship: {str(e)}, relationship data: {rel}, skipping.")

    return knowledge_triples


def save_extraction_results(kg_results: List[Dict],
                            output_kg_path: str,
                            output_triple_path: str) -> int:
    """
        Save the knowledge graph extraction results and extracted triples to JSON files.

        Args:
            kg_results (List[Dict]): A list of complete knowledge graph extraction results.
            output_kg_path (str): The file path for saving the knowledge graph results.
            output_triple_path (str): The file path for saving the triple results.

        Returns:
            int: The number of extracted triples.
    """
    triples = extract_triples_from_kg_results(kg_results)

    with open(output_kg_path, "w", encoding="utf-8") as f:
        json.dump(kg_results, f, ensure_ascii=False, indent=2)
    print(f"`The knowledge graph results have been saved to: {output_kg_path}`")

    with open(output_triple_path, "w", encoding="utf-8") as f:
        json.dump(triples, f, ensure_ascii=False, indent=2)
    print(f"The triplet results have been saved to: {output_triple_path}")

    return len(triples)


# --- Main Process Function ---

def run_kg_extraction_pipeline(
        document_file: str,
        kg_output_file: str,
        triple_output_file: str,
        redis_config: Dict,
        llm_config: Dict,
        dataset_config: Dict,
        running_config: Dict
) -> Dict:
    """
        Execute the complete knowledge graph extraction process.
        Args:
        document_file (str): Path to the input document file.
        kg_output_file (str): Path to the output file for knowledge graph results.
        triple_output_file (str): Path to the output file for triple results.
        redis_config (Dict): Redis connection configuration.
        llm_config (Dict): LLM client configuration (e.g., endpoints).
        dataset_config(Dict) : Dataset-related configurations(e.g., document_limit).
        running _config(Dict) : Runtime configurations(e.g., use_multithreading,max_workers).
        Returns:
        Dict: Statistics including counts of successful/failed documents and total number of triples generated.
    """
    print("Initializing components...")
    redis_client = redis.Redis(**redis_config)
    llm_client = LLMClient(endpoints=llm_config['endpoints'], client_cfg=ClientConfig(cache=redis_client))
    print("Loading and chunking documents...")
    # 传递分块参数
    chunking_params = {
        'overlap': running_config.get('overlap_size', 16),  # 使用 .get() 提供默认值
        'max_chunk_size': running_config.get('max_chunk_size', 128),
        'min_chunk_size': running_config.get('min_chunk_size', 100),
        'padding': running_config.get('padding_str', " ...")
    }
    documents = load_and_chunk_documents(document_file, dataset_config['document_limit'], chunking_params)

    print(f"Starting processing {len(documents)} document chunks...")
    kg_results = process_documents_batch(
        documents,
        llm_client,
        running_config['use_multithreading'],
        running_config['max_workers']
    )

    print("Saving...")
    n_of_triples = save_extraction_results(kg_results, kg_output_file, triple_output_file)

    success_count = sum(1 for r in kg_results if r and r.get("status") == "success")
    failed_count = len(kg_results) - success_count

    print(f"Processing completed. Success: {success_count}/{len(kg_results)}, Triples extracted: {n_of_triples}")
    print(f'Token Cost: {configs.TOKEN_COUNT}')

    return {
        'success_chunks': success_count,
        'extracted_triples_count': n_of_triples,
        'failed_chunks': failed_count,
        'total_chunks': len(kg_results)
    }


# --- Application Entry Point ---

def dataset_to_graph():
    """The main entry function, used for multiple attempts at knowledge graph extraction until all documents are successfully processed or the maximum number of retries is reached."""
    max_retries = 5
    for i in range(max_retries):
        print(f"\n--- Attempt {i + 1} at Knowledge Graph Extraction ---")
        results = run_kg_extraction_pipeline(
            document_file=DOCUMENT_FILE,
            kg_output_file=KG_OUTPUT_FILE,
            triple_output_file=TRIPLE_OUTPUT_FILE,
            redis_config=configs.REDIS_CONFIG,
            llm_config=configs.LLM_CONFIG,
            dataset_config=configs.DATASET_CONFIG,
            running_config=configs.RUNNING_CONFIG
        )
        if results['failed_chunks'] == 0:
            print("All document chunks have been successfully processed!")
            break
        else:
            print(f"There are still {results['failed_chunks']} document chunks that failed to process, retrying...")
        print(f"Current cumulative token consumption: {configs.TOKEN_COUNT}")
    else:
        print(f"\nMaximum retry count {max_retries} reached, some document chunks still remain unprocessed.")



if __name__ == "__main__":
    dataset_to_graph()