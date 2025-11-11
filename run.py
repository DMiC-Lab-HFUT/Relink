import configs
from core.KG_extraction import dataset_to_graph, test_document_processing
from add_graph_to_neo4j_threaded import import_triples_to_neo4j
from build_co_occur_graph import build_co_occur_graph
from embedding_documents import documents_to_vector_store
from embedding_entities import entities_to_vector_store
import json
from tqdm import tqdm
from main import run
from utils.neo4j_operator import deduplicates_list

def prepare_documents():
    with open(configs.DATASET_CONFIG['dataset_file'], 'r', encoding='utf-8') as f:
        samples = json.load(f)

    documents = []
    for sample in tqdm(samples):
        for title, texts in sample['context']:
            text = "\n".join(texts)
            d = {
                'label': configs.DATASET_CONFIG['domain'],
                'title': title,
                'text': text,
                'meta': {'Q': sample['question'], 'title': title, 'chunks': texts},
            }
            documents.append(d)
    # documents = list({d['text']: d for d in documents}.values())
    with open(configs.DATASET_CONFIG['document_file'], 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

def main():
    print("Preparing documents...")
    prepare_documents()
    print("Extracting triples from documents")
    dataset_to_graph()
    print("importing triples to neo4j")
    import_triples_to_neo4j()
    print("building co-occur graph and importing to neo4j")
    build_co_occur_graph()
    print("building documents vector store")
    documents_to_vector_store()
    print("building entities to vector store")
    entities_to_vector_store()
    print("running queries")
    for i in range(1):
        run(epoch=i)
        print(f"Current maximum depth: {configs.RETRIEVER_CONFIG['max_depth']}， Maximum breadth: {configs.RETRIEVER_CONFIG['max_width']}")
        print(configs.TOKEN_COUNT)

if __name__ == '__main__':

    main()