import json
from typing import List, Tuple, Dict, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from functools import partial
from utils.neo4j_operator import *
import configs
import random
random.seed(9527)
# 常量配置
BATCH_SIZE = 1000
MAX_WORKERS = 40

def load_triples(path: str) -> List[Dict[str, Any]]:
    """加载三元组数据"""
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def preprocess_triple(triple: Dict[str, Any], global_label: str) -> Tuple[List[Tuple[str, Dict]], List[Dict]]:
    """
    预处理三元组，生成实体和关系
    返回：[(label, entity)], [relation]
    """
    subject, obj = triple['subject'], triple['object']
    subject['id'] = hash_text(subject['mention'].lower().strip())
    obj['id'] = hash_text(obj['mention'].lower().strip())
    subject['name'], obj['name'] = subject['mention'], obj['mention']
    subject['source'] = obj['source'] = triple['meta']['title']

    entities, relations = [], []

    # Query-specific label
    if configs.DATASET_CONFIG['query_setting']:
        local_label = sanitize_label(triple['meta']['Q'])
        entities += [(local_label, subject), (local_label, obj)]
        relations.append({
            "head_id": subject['id'],
            "tail_id": obj['id'],
            "rel_type": sanitize_label(triple['relation']),
            "properties": {"evidence": triple['evidence'], 'title': triple['meta']['title']},
            "label": local_label
        })

    # Global label
    entities += [(global_label, subject), (global_label, obj)]
    relations.append({
        "head_id": subject['id'],
        "tail_id": obj['id'],
        "rel_type": triple['relation'],
        "properties": {"evidence": triple['evidence'], 'title': triple['meta']['title']},
        "label": global_label
    })

    return entities, relations

def group_and_merge_entities(entities: List[Tuple[str, Dict]]) -> Dict[str, Dict[str, Dict]]:
    """按标签和id分组并合并实体属性"""
    grouped = defaultdict(dict)
    for label, entity in entities:
        eid = entity['id']
        if eid in grouped[label]:
            grouped[label][eid] = merge_dicts(entity, grouped[label][eid])
        else:
            grouped[label][eid] = entity.copy()
    return grouped

def batch_insert_entities(neo: Neo4jOperations, grouped_entities: Dict[str, Dict[str, Dict]], batch_size: int, max_workers: int):
    """批量插入实体到Neo4j"""
    entity_batches = []
    for label, ents in grouped_entities.items():
        ents_list = list(ents.values())
        for i in range(0, len(ents_list), batch_size):
            batch = ents_list[i:i + batch_size]
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
    rel_grouped = defaultdict(list)
    for relation in relations:
        rel_grouped[relation['label']].append(relation)

    relation_batches = []
    for label, rels in rel_grouped.items():
        for i in range(0, len(rels), batch_size):
            batch = rels[i:i + batch_size]
            relation_batches.append((label, batch))

    total_relations = sum(len(batch) for _, batch in relation_batches)

    with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=total_relations, desc="Importing relations") as pbar:
        futures = [
            executor.submit(
                partial(
                    neo.batch_create_relationships_apoc_dynamic,
                    head_label=label,
                    tail_label=label,  # 可根据实际需要调整
                    relationships=batch
                )
            )
            for label, batch in relation_batches
        ]
        for future in as_completed(futures):
            try:
                inserted_count = future.result()
                pbar.update(inserted_count)
            except Exception as e:
                print(f"[Relation Insert Error]: {e}")


def import_triples_to_neo4j(
        triples_path: str = f"{configs.DATASET_CONFIG['dataset_path']}/output/extracted_triples.json",
        neo4j_uri: str = configs.NEO4J_CONFIG['uri'],
        neo4j_auth: tuple = configs.NEO4J_CONFIG['auth'],
        global_label: str = configs.DATASET_CONFIG['domain'],
        max_workers: int = MAX_WORKERS,
        batch_size: int = BATCH_SIZE
):

    print(f"importing Dataset ###{configs.DATASET_CONFIG['dataset_name']}###")
    """主流程：导入三元组到Neo4j"""
    triples = load_triples(triples_path)
    neo = Neo4jOperations(uri=neo4j_uri, auth=neo4j_auth)

    entity_tuples, relation_list = [], []
    for triple in tqdm(triples, desc="Preprocessing triples"):
        entities, relations = preprocess_triple(triple, global_label)
        entity_tuples.extend(entities)
        relation_list.extend(relations)

    print(f"Entities before deduplication: {len(entity_tuples)}")
    print(f"Relations before deduplication: {len(relation_list)}")
    entity_tuples = deduplicates_list(entity_tuples)
    relation_list = deduplicates_list(relation_list)
    print(f"Entities after deduplication: {len(entity_tuples)}")
    print(f"Relations after deduplication: {len(relation_list)}")


    #清理数据库
    neo.clean_database()
    # 添加到数据库中
    grouped_entities = group_and_merge_entities(entity_tuples)
    batch_insert_entities(neo, grouped_entities, batch_size, max_workers)

    neo.create_id_index(configs.DATASET_CONFIG['domain'])
    for label in tqdm(set([tup[0] for tup in entity_tuples]), desc="Creating index"):
        neo.create_id_index(label)

    relation_list = random.sample(relation_list, int(len(relation_list)))
    batch_insert_relations(neo, relation_list, batch_size, max_workers)


if __name__ == "__main__":
    import_triples_to_neo4j()