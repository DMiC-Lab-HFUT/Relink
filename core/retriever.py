import json
import traceback
from typing import List, Dict, Any, Optional
from utils.embeddings import CustomEmbedding
from utils.neo4j_operator import Neo4jOperations, sanitize_label
from utils.vector_store import MyVectorStore
from utils.my_llm import LLMClient, ClientConfig
from core.llm_functions import (
    topic_entity_extraction,
    eval_sufficiency_with_llm,
    complete_relations
)
import hashlib
import redis
from configs import *
from core.path_ranker import PathRanker
from collections import defaultdict
# from memory_profiler import profile
from line_profiler_pycharm import profile
import copy
from tabulate import tabulate
from typing import List, Set
from copy import deepcopy # 明确导入 deepcopy

class GraphRetriever:
    def __init__(self):
        """图数据库检索器初始化"""
        self.init_config()

    def init_config(self):
        # 图数据库
        self.graph_client = Neo4jOperations(
            uri=NEO4J_CONFIG["uri"],
            auth=NEO4J_CONFIG["auth"]
        )

        # 共现图谱
        if PIPLINE_CONFIG['expand_COG']:
            self.co_occur_graph = Neo4jOperations(**OCCURRENCE_GRAPH)

        #  redis缓存
        self.redis_client = redis.Redis(**REDIS_CONFIG)
        self.llm = LLMClient(endpoints=LLM_CONFIG['endpoints'], client_cfg=ClientConfig(cache=self.redis_client))

        # 嵌入模型
        self.embedding_model = CustomEmbedding(
            api_key=EMBEDDING_CONFIG['api_key'],
            base_url=EMBEDDING_CONFIG['base_url'],
            model_name=EMBEDDING_CONFIG['model_name']
        )

        # 实体嵌入数据库
        self.entities_vector_db = MyVectorStore(
            DATASET_CONFIG['entities_vector_store_path'],
            embedding=self.embedding_model
        )

        self.doc_vector_db = MyVectorStore(
            DATASET_CONFIG['doc_vector_store_path'],
            embedding=self.embedding_model
        )

        self.ranker = PathRanker(
            llm_client=self.llm,
            embedding_model=self.embedding_model,
            strategy=RETRIEVER_CONFIG.get("rank_strategy", "embedding"))

        # 运行配置参数
        self.sufficiency_check = RETRIEVER_CONFIG.get("sufficiency_check", False)
        self.min_similarity = RETRIEVER_CONFIG.get("min_similarity", 0.6)

    def retrieve(self, query: str, domain: str, max_depth: int = 3, max_width: int = 3) -> List[Dict[str, Any]]:
        """
        图数据库检索主方法
        Args:
            query: 用户查询
            max_depth: 搜索的最大深度
        Returns:
            检索结果路径列表，按相关性排序
        """
        logger.info(f"开始检索查询: {query}")
        try:
            start_entities = self.preprocess_query(query, domain)
            logger.info(f"start_entities: {[e['mention'] for e in start_entities]}")
            if not start_entities:
                logger.error("未识别到有效实体")
                return []

            # 使用beam search 检索与查询最为相关的路径
            results = self._beam_search(query, domain, start_entities, max_depth=max_depth, max_width=max_width)

            # 检索路径中三元组相关的句子作为补充上下文
            if PIPLINE_CONFIG["TCR"]:
                results = self.triple_context_restoration(results, query)

            temp_triples = list(set([
                f"<{triple['begin']['mention']} | {triple['r']} | {triple['end']['mention']}>"
                for path in results for triple in path.get('relations', [])
            ]))
            TRIPLE_COUNT['latent'] += len([r for r in temp_triples if r.split('|')[1].strip().startswith('_')])
            TRIPLE_COUNT['total'] += len(temp_triples)
            # print(TRIPLE_COUNT)

            return sorted(results, key=lambda x: x['score'], reverse=True)

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"检索过程异常: {tb}")
            return []

    def get_source(self, ent):
        if ent.get('source_list'):
            return list(set(ent['source_list'] + [ent['source']]))
        elif ent.get('source'):
            return [ent['source']]
        else:
            return []

    def triple_context_restoration(self, paths, query, k=1, retrieve_from_source=True):
        cache_key = f"RELINK|TCR:{str(paths)}"
        cached = self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
        for path in paths:
            context_sentences = []
            for relation in path['relations']:
                format_rel = f"{relation['begin']['mention']}, {relation['r'].replace('_', ' ')}, {relation['end']['mention']}"
                sources = self.get_source(relation['begin']) + self.get_source(relation['end'])
                if len(sources) != 0:
                    docs = self.doc_vector_db.query_from_collections([sanitize_label(s) for s in sources], query, k)
                else:
                    docs = self.doc_vector_db.query_collection(DATASET_CONFIG['domain'], format_rel)
                context_sentences.extend(
                    [doc.page_content for doc in docs[:k]]
                )
            path['context_sentences'] = context_sentences
        # paths[0]['context_sentences'] += sent_related_query

        self.redis_client.set(cache_key, json.dumps(paths))
        return paths

    def preprocess_query(self, query: str, domain: str) -> List[Dict]:
        """
        查询预处理（仅缓存向量查询部分）
        """
        try:
            query = query.strip()
            # 1. 主题实体提取（此部分不缓存，每次都执行）
            topic_entities = topic_entity_extraction(self.llm, query)

            contents = [
                f"{e['mention']} | {e.get('description', '')}"
                for e in topic_entities
            ]

            final_metadata = []
            # 2. 对每个内容进行向量查询，并应用缓存逻辑
            for content in contents:
                # 为每个独立的向量查询创建一个唯一的缓存键
                cache_key = f"vector_query_cache:{domain}:{content}"

                try:
                    # 尝试从 Redis 获取缓存
                    cached_result = self.redis_client.get(cache_key)
                    if cached_result:
                        # 缓存命中，直接使用缓存的元数据
                        logger.debug(f"向量查询缓存命中: {cache_key[:100]}...")
                        final_metadata.append(json.loads(cached_result))
                        continue  # 继续处理下一个 content

                except Exception as e:
                    logger.debug(f"Redis 缓存读取异常: {str(e)}")

                # --- 缓存未命中，执行实际的向量查询 ---
                logger.debug(f"向量查询缓存未命中，执行查询: {cache_key[:100]}...")

                # 执行耗时的向量数据库查询
                result = self.entities_vector_db.query_collection(
                    domain,
                    content
                )

                # 提取所需的结果（top-1 的元数据）
                metadata = result[0].metadata
                final_metadata.append(metadata)

                # 将新结果存入 Redis 缓存以备后用
                try:
                    self.redis_client.set(cache_key, json.dumps(metadata))  # 缓存1小时
                    logger.info(f"向量查询结果已存入缓存: {cache_key[:100]}...")
                except Exception as e:
                    logger.error(f"Redis 缓存写入异常: {str(e)}")

            return final_metadata

        except Exception as e:
            logger.error(f"预处理主流程异常: {str(e)}")
            return []
    def shortest_path(self, query, answer, domain):
        """
        获取从查询到答案的最短路径
        :param query:
        :param answer:
        :return: 返回不为空的路径列表
        """
        start_entities = self.preprocess_query(query, domain=domain)
        query = query.strip()
        query_domain = sanitize_label(query)
        end_entity = self._link_to_graph(answer, query_domain)

        # 获取所有路径
        paths = [
            self.graph_client.find_shortest_path(
                head_id=ent['id'],
                tail_id=end_entity['id'],
                head_label=query_domain,
                tail_label=query_domain
            ) for ent in start_entities
        ]

        # 过滤掉为空的路径
        non_empty_paths = [path for path in paths if path]

        # 返回不为空的路径
        return [{'relations': path} for path in non_empty_paths]

    def _link_to_graph(self, entity, question_domain):
        query_text = f"{entity}"
        results = self.entities_vector_db.query_collection(
            question_domain,  # 标准化问题标签
            query_text
        )
        return results[0].metadata

    def output_log(self, final_depth, current_paths, query, level='info'):
        # 公共部分
        logger_func = getattr(logger, level, logger.info)  # 支持 info/debug/warning 等
        path_num = len(current_paths)

        if not current_paths:
            logger_func("当前没有可用路径。")
            return
        # 表格输出
        rows = [
            (p.format_string, str([round(s, 3) for s in p.scores]), round(p.score, 3))
            for p in current_paths
        ]
        headers = ("Path", "Scores", "Score")
        output = tabulate(
            rows,
            headers=headers,
            tablefmt="pretty",
            colalign=("left", "left", "left")
        )
        logger_func(f"\nDepth: {final_depth}, Path Num: {path_num}")
        logger_func(f"\nQuery: {query}\nCurrent paths:\n{output}\n{'=' * 40}")

    @profile
    def _beam_search(
            self,
            query: str,
            domain: str,
            initial_entities: List[Dict],
            max_depth: int,
            max_width: int,
            prune_zero_score: bool = False  # 控制是否剪枝分数为0的新路径
    ) -> List[Dict]:
        """带剪枝的宽度优先搜索，保留走到头的路径，走到头的不再打分"""
        current_paths = [Path(e) for e in initial_entities]

        for depth in range(max_depth):
            logger.debug(f"Executing Beam Search Depth: {depth}")
            # 1. 区分已完成和需扩展路径
            finished_paths = [p for p in current_paths if p.is_finished]
            to_expand_paths = [p for p in current_paths if not p.is_finished]

            # 2. 扩展新路径
            cooccur_expanded_paths = []
            fact_expanded_paths = []

            for path in to_expand_paths:
                logger.debug(f"expand from knowledge graph: {domain}")
                kg_expanded_paths = self.expand_from_knowledge_graph(path, domain)
                fact_expanded_paths.extend(kg_expanded_paths)
                if PIPLINE_CONFIG['expand_COG']:
                    logger.debug(f"expand from co-occurrence graph: domain: {domain} | id: {path.current_node['id']}")
                    kg_new_node_ids = {p.current_node['id'] for p in kg_expanded_paths}
                    cooccur_expanded_paths.extend(self.expand_from_cooccurrence_graph(path, domain, exclude_ids=kg_new_node_ids))
                    logger.debug(f"Current co-occurrence paths: {len(cooccur_expanded_paths)}")

                if not fact_expanded_paths and not cooccur_expanded_paths:
                    path.is_finished = True
                    finished_paths.append(path)

            all_paths = []

            # 3. 新路径统一打分和剪枝
            if fact_expanded_paths:
                logger.debug(f"scored expanded fact paths: {len(fact_expanded_paths)}")
                new_paths = list({p.format_string: p for p in fact_expanded_paths}.values())
                scored_fact_expanded_paths = self.ranker.score_and_rank_paths(new_paths, query)
                all_paths += scored_fact_expanded_paths

                # 保存路径数据用于训练
                path_saved = [(query, path) for query, path in
                                       zip([query] * len(scored_fact_expanded_paths), [p.to_dict() for p in scored_fact_expanded_paths])]
                for p in path_saved:
                    write_to_file(json.dumps(p))


            if PIPLINE_CONFIG['expand_COG'] and cooccur_expanded_paths:
                logger.debug(f"scored expanded co-occur paths: {len(cooccur_expanded_paths)}")
                new_paths = list({p.format_string: p for p in cooccur_expanded_paths}.values())
                scored_cooccur_expanded_paths = self.ranker.score_and_rank_paths(new_paths, query)
                cooccur_paths = scored_cooccur_expanded_paths[:40]
                logger.debug(f"covering co-occurrence to rel: {len(cooccur_paths)}")
                paths = [p for p in self.cover_co_occurrence_to_rel(cooccur_paths, query=query, remove_none=False) if "none" not in p.format_string]
                all_paths += paths

            if PIPLINE_CONFIG['two_stage'] and RETRIEVER_CONFIG['rank_strategy'] == "trained_ranker":
                all_paths = [p for p in self.ranker.score_and_rank_paths(all_paths, query, strategy='llm')]

            # 剪枝
            current_paths = sorted(all_paths, key=lambda p: p.score, reverse=True)[:max_width]

            # 5. 日志
            self.output_log(depth + 1, current_paths, query, level='debug')

            # 6. 路径充分性检查
            if self.sufficiency_check and self._eval_sufficiency(current_paths, query):
                logger.debug("路径充分，提前终止")
                break

        # 8. 统计与最终日志
        if current_paths:
            final_depth = len(current_paths[0])
        else:
            final_depth = 0

        self.output_log(final_depth, current_paths, query)
        STATISTIC['n_of_path'] += 1
        STATISTIC['average_depth'] = ((STATISTIC['n_of_path'] - 1) * STATISTIC['average_depth'] + final_depth) / \
                                     STATISTIC['n_of_path']
        STATISTIC['path_lens'].append(final_depth)

        return [p.to_dict() for p in current_paths]



    def _expand_domain_edges(self, path: 'Path', domain) -> List['Path']:
        """通过领域边扩展路径"""
        try:
            triples = self.graph_client.get_connected_edges_and_nodes(
                path.current_node['id'],
                label=domain
            )
            expanded_paths = [
                path.copy().add_node(triple['target_node'], triple['edge'])
                for triple in triples
            ]
            return expanded_paths
        except Exception as e:
            logger.error(f"领域边路径扩展失败: {str(e)}")
            raise e

    def _expand_cooccurrence_edges(self, path: 'Path', exclude_ids: set, domain=None) -> List['Path']:
        """通过共现边扩展路径"""
        try:
            co_occur_triples = []
            for ent, doc in self.co_occur_graph.query_cooccurrence_entities_with_docs_degree_limit(path.current_node['id'], domain):
                if ent['id'] not in exclude_ids:
                    co_occur_triples.append({
                        'begin': path.current_node,
                        'end': ent,
                        'r': "co-occurrence",
                        'title': doc['title'],
                        "text": doc['text']
                    })

            expanded_paths = [
                path.copy().add_node(triple['end'], triple) for triple in co_occur_triples
            ]
            return expanded_paths
        except Exception as e:
            logger.error(f"共现边路径扩展失败: {str(e)}")
            return []

    def expand_from_knowledge_graph(self, path: 'Path', domain) -> List['Path']:
        """
        从知识图谱扩展路径。
        此函数封装了所有与知识图谱路径扩展相关的逻辑，如防止循环。
        """
        # 获取当前路径中已访问过的节点，防止循环
        visited_ids = {n['id'] for n in path.nodes}

        # 1. 使用辅助函数从知识图谱获取新的潜在路径
        new_paths = self._expand_domain_edges(path, domain)

        # 2. 过滤掉导致循环的路径
        valid_paths = [p for p in new_paths if p.current_node['id'] not in visited_ids]

        return valid_paths

    def expand_from_cooccurrence_graph(self, path: 'Path', domain, exclude_ids: Set[str]) -> List['Path']:
        """
        从共现图谱扩展路径。
        此函数封装了所有与共现路径扩展相关的逻辑。

        Args:
            path (Path): 需要扩展的原始路径。
            domain: 限制扩展的领域。
            exclude_ids (Set[str]): 在此次扩展中需要排除的节点ID集合，
                                     通常是已通过知识图谱扩展出的节点，避免重复。
        """
        # 获取当前路径中已访问过的节点，加上需要额外排除的节点
        visited_ids = {n['id'] for n in path.nodes}

        # 1. 使用辅助函数从共现图谱获取新的潜在路径
        # 将 `exclude_ids` 传递给底层函数，以提高查询效率
        new_paths = self._expand_cooccurrence_edges(path, exclude_ids, domain)

        # 2. 过滤掉导致循环的路径
        # （虽然底层函数已经过滤了 exclude_ids，但这里再检查一遍 visited_ids 以确保安全）
        valid_paths = [p for p in new_paths if p.current_node['id'] not in visited_ids]

        return valid_paths

    def cover_co_occurrence_to_rel(self, paths: List['Path'], query="", remove_none=False) -> List['Path']:
        """
        替换所有 'co-occurrence' 关系为 LLM 预测的实际关系 (修正版)
        :param paths: 路径列表，每个路径包含若干关系三元组
        :return: 替换后的路径列表
        """
        CO_OCCURRENCE_RELATION = 'co-occurrence'

        # 1. 收集所有 co-occurrence 三元组的引用
        cooccur_triples = [
            triple
            for path in paths
            for triple in path.relations
            if triple['r'] == CO_OCCURRENCE_RELATION
        ]
        if not cooccur_triples:
            return paths

        # 2. 按文档分组三元组，这里的 triple 是原始对象的引用
        doc2triples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for triple in cooccur_triples:
            # 确保三元组中有 'title' 字段
            if 'title' not in triple:
                raise ValueError("三元组中缺少 'title' 字段，无法按文档分组。")
            doc2triples[triple['title']].append(triple)

        # 3. 按文档调用LLM并直接修改三元组
        entity_pair_to_relations = defaultdict(list)
        for title, triples_in_doc in doc2triples.items():
            text = triples_in_doc[0]['text']
            entity_pairs = [
                [triple['begin']['mention'], triple['end']['mention']]
                for triple in triples_in_doc
            ]
            try:
                completed_triples = complete_relations(self.llm, text, entity_pairs, query)
            except Exception as e:
                raise RuntimeError(f"LLM 关系预测失败，文档: {title}") from e

            for completed_triple in completed_triples:
                key = (completed_triple['head'], completed_triple['tail'])
                entity_pair_to_relations[key].extend(completed_triple['relations'])

        final_paths = []
        for path in paths:
            # 使用一个列表来管理可能因关系分裂而产生的多个路径版本
            # 开始时，只有一条路径，即原始路径的深拷贝
            wip_paths = [deepcopy(path)]

            # 遍历路径中的每一个关系（使用索引）
            for r_index, relation in enumerate(path.relations):
                if relation['r'] != CO_OCCURRENCE_RELATION:
                    # 如果不是共现关系，跳过，继续检查下一个关系
                    continue

                # 如果是共现关系，则需要对当前所有正在处理的路径 (wip_paths) 进行扩展
                key = (relation['begin']['mention'], relation['end']['mention'])
                predicted_rels = entity_pair_to_relations.get(key)

                # 如果没有找到预测结果
                if not predicted_rels:
                    if remove_none:
                        # 如果要移除none，则当前 wip_paths 中所有路径都无法完成，直接清空
                        wip_paths = []
                    else:
                        # 否则，将这个关系标记为 "none"，并继续处理
                        for p in wip_paths:
                            p.relations[r_index]['r'] = "_none"
                    # 跳出循环，因为这个分支（有或没有 'none'）已经处理完毕
                    break

                # 如果找到了预测结果，对路径进行分裂/扩展
                predicted_rels = list(dict.fromkeys(entity_pair_to_relations.get(key)))

                expanded_paths = []
                for p in wip_paths:  # 遍历当前已有的路径版本
                    for i, rel in enumerate(predicted_rels):
                        # 第一个关系可以直接修改，后续关系需要创建新的拷贝
                        if i == 0:
                            p.relations[r_index]['r'] = "_" + rel
                            expanded_paths.append(p)
                        else:
                            new_path_version = deepcopy(p)
                            new_path_version.relations[r_index]['r'] = "_" + rel
                            expanded_paths.append(new_path_version)

                # 用扩展后的路径列表替换原来的列表
                wip_paths = expanded_paths

            # 当前原始 path 的所有共现关系都处理完毕后，
            # 将所有成功生成的路径版本添加到最终结果中
            final_paths.extend(wip_paths)

        if len(final_paths) != len(paths):
            s1 = [p.format_string for p in final_paths]
            s2 = [p.format_string for p in paths]
            s1, s2

        return final_paths


    def _eval_sufficiency(
            self,
            paths: List['Path'],
            query: str
    ):
        """
        使用大模型验证当前收集的路径是否足够回答问题。
        :param paths:
        :param query:
        :return:
        """
        res = eval_sufficiency_with_llm(llm=self.llm, query=query, paths=[p.to_dict() for p in paths])
        return res
class Path:
    """
        表示知识图谱中的检索路径

        属性：
        - nodes: 路径中的节点列表（按顺序存储）
        - relations: 路径中的关系列表（连接节点的边）
        - score: 路径的总体相关性得分
        - _diversity: 路径多样性缓存值

        方法：
        - add_node: 添加节点和关系到路径
        - copy: 创建路径的深拷贝
        - to_dict: 转换为字典格式
        """

    def __init__(self, start_node: Dict):
        self.nodes = [start_node]
        self.relations = []
        self._diversity = None


        self.is_finished = False
        self.scores = []

    @property
    def score(self) -> float:
        valid_scores = [score for score in self.scores if score != 0.0]
        if valid_scores:
            self._score = sum(valid_scores) / (len(valid_scores) + 1e-8) #+ 0.01 * getattr(self, 'diversity', 0)
        else:
            self._score = 0.0
        return round(self._score, 5)

    @property
    def current_node(self) -> Dict:
        """获取路径末尾节点"""
        return self.nodes[-1]

    @property
    def diversity(self) -> float:
        """计算路径多样性（基于节点类型分布的辛普森指数）"""
        # if self._diversity is None:
        type_counts = {}
        for n in self.nodes:
            t = n.get('type', 'Unknown')
            type_counts[t] = type_counts.get(t, 0) + 1
        self._diversity = 1 - sum((v / len(self.nodes)) ** 2 for v in type_counts.values())
        return self._diversity

    def add_node(self, node: Dict, relation: Optional[Dict] = None) -> 'Path':
        """
                扩展路径

                :param node: 新节点
                :param relation: 连接关系
                :return: 更新后的Path对象
                """
        self.nodes.append(node)
        if relation:
            self.relations.append(relation)
        return self

    def pop_node(self) -> 'Path':
        self.nodes.pop()
        self.relations.pop()
        return self

    def copy(self) -> 'Path':
        return copy.deepcopy(self)

    def to_dict(self) -> Dict:

        return {
            "nodes": self.nodes,
            "relations": self.relations,
            "score": round(self.score, 5),
            "diversity": round(self.diversity, 5),
            "format_string": self.format_string,
            'scores': self.scores
        }

    def __len__(self):
        return len(self.nodes)

    @property
    def format_string(self, path_format="link"):
        if path_format == "link":
            format_string = f"{self.nodes[0]['mention']}"
            for relation, node in zip(self.relations, self.nodes[1:]):
                if relation['end']['id'] == node['id']:
                    format_string += f" - {relation['r']} -> {node['mention']}"
                else:
                    format_string += f" <- {relation['r']} - {node['mention']}"

            return format_string
        elif path_format == "triplets":
            triplet_str = []
            for relation, node in zip(self.relations, self.nodes[1:]):
                triplet_str.append(f"({relation['begin']['mention']}, {relation['r']}, {relation['end']['mention']})")
            format_string = '\n'.join(triplet_str)
            return format_string



class TextRetriever:
    def __init__(self):
        """图数据库检索器初始化"""
        self.init_config()

    def init_config(self):
        #  redis缓存
        self.redis_client = redis.Redis(**REDIS_CONFIG)
        self.llm = LLMClient(
            endpoints=LLM_CONFIG['endpoints'],
            cache=self.redis_client
        )

        # 嵌入模型
        self.embedding_model = CustomEmbedding(
            api_key=EMBEDDING_CONFIG['api_key'],
            base_url=EMBEDDING_CONFIG['base_url'],
            model_name=EMBEDDING_CONFIG['model_name']
        )

        self.doc_vector_db = MyVectorStore(
            DATASET_CONFIG['doc_vector_store_path'],
            embedding=self.embedding_model
        )

    def retrieve(self,query, domain):
        docs = self.doc_vector_db.query_collection(domain, query, k=4)
        return [d.page_content for d in docs]
