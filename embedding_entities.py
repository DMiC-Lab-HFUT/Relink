import json
import os
from collections import defaultdict
from tqdm import tqdm
from langchain_core.documents import Document
from utils.embeddings import CustomEmbedding
from utils.vector_store import MyVectorStore
from utils.neo4j_operator import sanitize_label, hash_text, deduplicates_list
import configs
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from tqdm import tqdm


def load_triples(file_path):
    try:
        with open(file_path, encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load triples from {file_path}: {e}")
        return None

def build_entity_list(triples):
    ents = []
    for triple in triples:
        triple['subject']['meta'] = triple['meta']
        triple['object']['meta'] = triple['meta']
        ents.append(triple['subject'])
        ents.append(triple['object'])
    return ents

def build_q2e_mapping(ents):
    q2e = defaultdict(list)
    for e in ents:
        e['id'] = hash_text(e['mention'].lower().strip())
        q2e[e['meta']['Q']].append(e)
    return q2e


def process_single_entity(q, ents, vector_store, domain):
    collection_name = sanitize_label(q)
    ents = deduplicates_list(ents)
    documents = [
        Document(page_content=f"{e['mention']} | {e.get('description', '')}", metadata=e)
        for e in ents
    ]
    try:
        embeddings = vector_store.embedding_documents(documents)
        vector_store.add_embedding_to_collection(sanitize_label(domain), documents, embeddings)
        vector_store.add_embedding_to_collection(collection_name, documents, embeddings)
        return collection_name, "success"
    except Exception as e:
        return collection_name, f"failed: {e}"

def store_entities_to_vector_store_threaded(q2e, vector_store, domain, max_workers=8):
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for q, ents in q2e.items():
            futures.append(executor.submit(process_single_entity, q, ents, vector_store, domain))
        # tqdm显示进度
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing entities (multi-thread)"):
            collection_name, status = f.result()
            if status != "success":
                print(f"Collection {collection_name}: {status}")
    vector_store.force_save_all()
    print(vector_store.get_performance_stats())


def store_entities_to_vector_store(q2e, vector_store, domain):
    for q, ents in tqdm(q2e.items(), desc="Processing entities"):
        collection_name = sanitize_label(q)
        ents = deduplicates_list(ents)
        documents = [
            Document(page_content=f"{e['mention']} | {e.get('description', '')}", metadata=e)
            for e in ents
        ]
        try:
            embeddings = vector_store.embedding_documents(documents)
            vector_store.add_embedding_to_collection(sanitize_label(domain), documents, embeddings)
            vector_store.add_embedding_to_collection(collection_name, documents, embeddings)
            # vector_store.add_to_collection(collection_name, documents)
            # vector_store.add_to_collection(sanitize_label(domain), documents)
        except Exception as e:
            print(f"Failed to add documents to collection {collection_name}: {e}")
    vector_store.force_save_all()
    # 查看性能统计
    print(vector_store.get_performance_stats())

def entities_to_vector_store():
    embeddings = CustomEmbedding(
        api_key=configs.EMBEDDING_CONFIG['api_key'],
        base_url=configs.EMBEDDING_CONFIG['base_url'],
        model_name=configs.EMBEDDING_CONFIG['model_name']
    )
    os.makedirs(configs.DATASET_CONFIG['entities_vector_store_path'], exist_ok=True)
    vector_store = MyVectorStore(configs.DATASET_CONFIG['entities_vector_store_path'], embeddings)
    triples = load_triples(configs.DATASET_CONFIG['kg_triples_file'])
    if triples is None:
        return
    # Construct the entity list and the mapping from queries to entities
    ents = build_entity_list(triples)
    q2e = build_q2e_mapping(ents)
    store_entities_to_vector_store_threaded(q2e, vector_store, domain=configs.DATASET_CONFIG['domain'])
    print(vector_store)

if __name__ == "__main__":
    entities_to_vector_store()
