
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from functools import partial
from utils.neo4j_operator import *
import configs

# 常量配置
BATCH_SIZE = 1000
MAX_WORKERS = 20

def load_json(filepath: str) -> Any:
    """加载JSON文件"""
    try:
        with open(filepath, encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"文件未找到: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"JSON解析错误: {filepath}")
        return None


def build_title2doc(documents: List[Dict]) -> Dict[str, str]:
    """构建标题到文档内容的映射"""
    return {doc['title']: doc['text'] for doc in documents}


def extract_co_occur_entities(triples: List[Dict], title2doc) -> Dict[str, List[Dict]]:
    """提取每个文档标题下的共现实体（去重）"""
    docs = dict()
    for triple in triples:
        label = sanitize_label(triple['meta']['Q'])
        title = triple['meta']['title']
        if docs.get(title) is None:
            docs[title] = {
                "entities": [],
                "title": title,
                'local_label': label,
                'global_label': configs.DATASET_CONFIG['domain'],
                'text': title2doc[title]
            }
        docs[title]['entities'].extend([triple['object'], triple['subject']])
    # 对每个标题下的实体列表进行去重（以实体mention为唯一标识）
    for doc in docs.values():
        unique = {}
        for ent in doc['entities']:
            mention = ent['mention']
            if mention not in unique:
                unique[mention] = ent
        doc['entities'] = list(unique.values())
    return docs


def get_docs():
    print(f"importing Dataset ###{configs.DATASET_CONFIG['dataset_name']}###")
    triples = load_json(f'{configs.DATASET_CONFIG["output_dir"]}/extracted_triples.json')
    documents = load_json(f'{configs.DATASET_CONFIG["document_file"]}')
    if not triples or not documents:
        return
    title2doc = build_title2doc(documents)
    docs = extract_co_occur_entities(triples, title2doc)
    return docs


def deduplicate_label_and_id(nodes):
    seen = set()
    result = []
    for node in nodes:
        key_tuple = (node[0], node[1]['id'])
        if key_tuple not in seen:
            seen.add(key_tuple)
            result.append(node)
    return result

def prepare_entities_relations(docs):
    entities = []
    doc_ents = []
    relations = []

    for doc in tqdm(docs.values()):
        global_label = doc['global_label']
        local_label = doc['local_label']
        doc_node = {
            "id": hash_text(doc['title'].lower().strip()),
            "title": doc['title'],
            'text': doc['text'],
        }
        doc_ents.append((f"Doc:{local_label}", doc_node))
        doc_ents.append((f"Doc:{global_label}", doc_node))

        for ent in doc['entities']:
            ent['id'] = hash_text(ent['mention'].lower().strip())
            entities.append((f"Entity:{local_label}", ent))
            entities.append((f"Entity:{global_label}", ent))
            relations.append({
                "head_label": f"Doc:{local_label}",
                "head_id": doc_node['id'],
                "tail_label": f"Entity:{local_label}",
                "tail_id": ent['id'],
                "rel_type": "contain"
            })

            relations.append({
                "head_label": f"Doc:{global_label}",
                "head_id": doc_node['id'],
                "tail_label": f"Entity:{global_label}",
                "tail_id": ent['id'],
                "rel_type": "contain"
            })
    nodes = entities + doc_ents
    nodes = deduplicate_label_and_id(nodes)
    return nodes, relations

def batch_insert_entities(neo: Neo4jOperations, nodes, batch_size=BATCH_SIZE, max_workers=MAX_WORKERS):
    """批量插入实体到Neo4j"""

    grouped_nodes = defaultdict(list)
    nodes = deduplicates_list(nodes)
    for node in nodes:
        grouped_nodes[node[0]].append(node[1])

    entity_batches = []
    for label, nodes in grouped_nodes.items():
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            entity_batches.append((label, batch))

    total_entities = sum(len(batch) for _, batch in entity_batches)

    with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=total_entities, desc="Importing entities") as pbar:
        futures = [
            executor.submit(partial(neo.batch_create_nodes_with_apoc, label, batch))
            for label, batch in entity_batches
        ]
        for future in as_completed(futures):
            try:
                inserted_count = future.result()
                pbar.update(inserted_count)
            except Exception as e:
                print(f"[Entity Insert Error]: {e}")

def batch_insert_relations(neo: Neo4jOperations, relations: List[Dict], batch_size: int, max_workers: int):
    """批量插入关系到Neo4j"""

    relations = deduplicates_list(relations)
    grouped_edges = defaultdict(list)
    for edge in relations:
        group = (edge['head_label'], edge['tail_label'], edge['rel_type'])
        grouped_edges[group].append(edge)

    relation_batches = []
    for (head_label, tail_label, rel_type), rels in grouped_edges.items():
        for i in range(0, len(rels), batch_size):
            batch = rels[i:i + batch_size]
            relation_batches.append(((head_label, tail_label, rel_type), batch))

    total_relations = sum(len(batch) for _, batch in relation_batches)

    with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=total_relations, desc="Importing relations") as pbar:
        futures = [
            executor.submit(
                partial(
                    neo.batch_create_relationships_apoc_dynamic,
                    head_label=head_label,
                    tail_label=tail_label,
                    relationships=batch
                )
            )
            for (head_label, tail_label, rel_type), batch in relation_batches
        ]
        for future in as_completed(futures):
            try:
                inserted_count = future.result()
                pbar.update(inserted_count)
            except Exception as e:
                print(f"[Relation Insert Error]: {e}")


def build_co_occur_graph():
    docs = get_docs()
    neo = Neo4jOperations(**configs.OCCURRENCE_GRAPH)  # 获取参数
    nodes, edges = prepare_entities_relations(docs)
    batch_insert_entities(neo, nodes, batch_size=BATCH_SIZE, max_workers=10)
    labels = set()
    for node in nodes:
        labels.update(node[0].split(":"))
    for label in tqdm(labels, desc="Building co_occur_graph index"):
        neo.create_id_index(label)
    batch_insert_relations(neo, edges, BATCH_SIZE, max_workers=20)


if __name__ == "__main__":
    build_co_occur_graph()


