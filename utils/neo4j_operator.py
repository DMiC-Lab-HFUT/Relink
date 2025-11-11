from neo4j import GraphDatabase
import traceback
import hashlib
import re
from typing import Dict, List, Optional
import json
from typing import List, Any
from neo4j import Driver, exceptions as neo4j_exceptions
from typing import Optional, Tuple, List
from neo4j.exceptions import Neo4jError
from collections import defaultdict
from neo4j.exceptions import Neo4jError
from tqdm import tqdm

def deduplicates_list(lst: List[Any]) -> List[Any]:
    """
    去除列表中的重复元素，并保持原有顺序。

    参数:
    lst (List[Any]): 需要去重的列表，列表中的元素可以是任何可序列化的类型。

    返回:
    List[Any]: 去重后的列表，保持原有顺序。
    """
    if not isinstance(lst, list):
        raise TypeError("输入参数必须是一个列表")

    unique_dict = {}
    for item in lst:
        # 将每个元素序列化为字符串作为键，确保字典的键是唯一的
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key not in unique_dict:
            unique_dict[key] = item

    # 返回去重后的列表
    return list(unique_dict.values())


def hash_string(input_string: str) -> str:
    """使用 MD5 生成哈希字符串并截取前8位

    Args:
        input_string: 输入字符串

    Returns:
        8位十六进制哈希值
    """
    return hashlib.md5(input_string.encode()).hexdigest()[:8]


def hash_text(input_string: str) -> str:
    """使用 MD5 生成哈希字符串并截取后15位

    Args:
        input_string: 输入字符串

    Returns:
        15位十六进制哈希值
    """
    return hashlib.md5(input_string.encode()).hexdigest()[-15:]


def levenshtein_distance(s1: str, s2: str) -> int:
    """计算两个字符串的编辑距离(Levenshtein Distance)

    Args:
        s1: 字符串1
        s2: 字符串2

    Returns:
        最小编辑操作次数
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """计算基于编辑距离的相似度(1 - 标准化编辑距离)

    Args:
        s1: 字符串1
        s2: 字符串2

    Returns:
        相似度值，范围[0,1]
    """
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0  # 处理空字符串情况
    return 1 - levenshtein_distance(s1, s2) / max_len


def sanitize_label(label: str) -> str:
    """清洗节点标签使其符合Neo4j规范

    1. 替换非法字符为下划线
    2. 处理数字开头情况（在数字前添加下划线）
    3. 去除多余下划线

    Args:
        label: 原始标签

    Returns:
        合法标签

    Raises:
        AssertionError: 标签为空或处理后为空
    """
    # 替换非字母数字下划线的字符
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', label)

    # 去除连续下划线
    sanitized = re.sub(r'__+', '_', sanitized).strip('_')

    # 处理开头数字（在数字前添加下划线）
    if sanitized and sanitized[0].isdigit():
        sanitized = '_' + sanitized
    # 验证结果
    assert len(sanitized) > 0, "标签不能为空"
    return sanitized


def merge_dicts(new_dict: Dict, old_dict: Dict,
                excluded_keys: Optional[List] = None,
                postfix: str = '_list') -> Dict:
    """合并新旧节点属性字典

    Args:
        new_dict: 新属性字典
        old_dict: 旧属性字典
        excluded_keys: 需要排除的键
        postfix: 列表属性后缀

    Returns:
        合并后的属性字典
    """
    excluded = excluded_keys or ['id', 'labels']
    merged = new_dict.copy()

    for key, old_val in old_dict.items():
        if key in excluded:
            continue

        new_val = new_dict.get(key)

        if new_val is not None:
            if new_val != old_val:
                # 处理列表类型属性
                if key.endswith(postfix) and isinstance(new_val, list):
                    merged[key] = list(set(new_val + old_val))
                else:
                    # 创建新列表属性
                    list_key = f"{key}{postfix}"
                    merged[list_key] = list(set(
                        new_dict.get(list_key, []) +
                        old_dict.get(list_key, []) +
                        [new_val, old_val]
                    ))
        else:
            merged[key] = old_val

    return merged


class Neo4jOperations:
    """Neo4j 数据库操作封装类"""

    def __init__(self, uri: str = "bolt://localhost:7687",
                 auth: tuple = ("neo4j", "12345678")):
        """初始化数据库连接

        Args:
            uri: 数据库连接URI
            auth: 认证信息(用户名, 密码)
        """
        self.driver = GraphDatabase.driver(uri, auth=auth)
        try:
            self.driver.verify_connectivity()
        except Exception as e:
            raise ConnectionError(f"无法连接到Neo4j数据库:{uri}: {auth}") from e

    def close(self) -> None:
        """关闭数据库连接"""
        self.driver.close()

    def create_or_update_node(self, label: str, properties: Dict) -> None:
        """创建或更新节点

        Args:
            label: 已清洗的节点标签
            properties: 节点属性字典(必须包含id字段)
        """
        node_id = properties.get('id')
        if not node_id:
            raise ValueError("节点必须包含id属性")

        label = label
        cypher = f"""
        MERGE (n:{label} {{id: $id}})
        ON CREATE SET n += $props
        ON MATCH SET n += $props
        """

        with self.driver.session() as session:
            session.execute_write(
                lambda tx: tx.run(cypher, id=node_id, props=properties)
            )

    def batch_create_or_update_nodes(self, label: str, properties_list: list, batch_size: int = 10000) -> None:
        """
        批量创建或更新节点

        Args:
            label: 已清洗的节点标签
            properties_list: 节点属性字典组成的列表（每个dict必须包含id字段）
            batch_size: 每批次插入的数量，默认1000
        """
        if not properties_list:
            return

        label = sanitize_label(label)
        cypher = f"""
        UNWIND $batch AS row
        MERGE (n:{label} {{id: row.id}})
        SET n += row
        """

        def chunked_iterable(iterable, size):
            for i in range(0, len(iterable), size):
                yield iterable[i:i + size]

        with self.driver.session() as session:
            for batch in tqdm(chunked_iterable(properties_list, batch_size), desc="Adding entities by batch"):
                session.execute_write(
                    lambda tx: tx.run(cypher, batch=batch)
                )

    def batch_create_or_update_nodes_apoc(self, label: str, properties_list: list, batch_size: int = 10000) -> None:
        """
        【APOC 加速版】批量创建或更新节点

        Args:
            label: 已清洗的节点标签
            properties_list: 节点属性字典组成的列表（每个dict必须包含id字段）
            batch_size: 每批次处理的数量，默认10000
        """
        if not properties_list:
            return

        label = sanitize_label(label)  # 清洗标签的操作保留

        # 使用 apoc.periodic.iterate 的 Cypher 查询语句
        # 注意：我们将整个列表作为单个参数 $properties 传递
        cypher = f"""
        CALL apoc.periodic.iterate(
            "UNWIND $properties AS row RETURN row",  // 第一个语句：告诉 APOC 要遍历的数据源
            "MERGE (n:{label} {{id: row.id}}) SET n += row",  // 第二个语句：对每批数据执行的操作
            {{batchSize: {batch_size}, parallel: true, params: {{properties: $properties}}}} // 配置项
        )
        """
        with self.driver.session() as session:
            # 不再需要 tqdm 和手动分块循环，直接执行一次查询
            # 将整个 properties_list 作为参数传递
            session.run(cypher, properties=properties_list)

    def batch_create_nodes_with_apoc(self, label: str, properties_list: list, max_entities: int = 100_000) -> int:
        """
        批量创建或更新节点（单次插入，超过10万报错）

        Args:
            label: 已清洗的节点标签
            properties_list: 节点属性字典组成的列表（每个dict必须包含id字段）
            max_entities: 最大允许插入数量，默认10万

        Returns:
            实际插入的实体数量
        """
        if not properties_list:
            return 0

        total = len(properties_list)
        if total > max_entities:
            raise ValueError(f"单次插入的实体数量超过上限：{max_entities}，实际为：{total}")

        label = label
        cypher = f"""
           UNWIND $batch AS row
           CREATE (n:{label})
           SET n += row
           """

        with self.driver.session() as session:
            session.execute_write(
                lambda tx: tx.run(cypher, batch=properties_list)
            )

        return total

    def batch_create_relationships_apoc_dynamic(
            self, head_label, tail_label, relationships, batch_size=1000, data_limit=100_000
    ):
        """
        用 APOC 创建动态类型关系，并返回实际插入数量

        :param head_label: 起始节点标签
        :param tail_label: 目标节点标签
        :param relationships: [
            {
                "head_id": ...,
                "tail_id": ...,
                "rel_type": ...,
                "properties": ...,
            },
            ...
        ]
        :param batch_size: 每批处理数量
        :param data_limit: 最大允许插入的关系数量
        :return: 实际插入的关系数量
        """
        total = len(relationships)
        if not relationships:
            return 0

        if total > data_limit:
            raise ValueError(f"单次插入关系数量 {total} 超过上限 {data_limit}，请分批处理！")

        cypher = f"""
            CALL apoc.periodic.iterate(
              "UNWIND $rows AS row RETURN row",
              "
                MATCH (a:{head_label} {{id: row.head_id}})
                MATCH (b:{tail_label} {{id: row.tail_id}})
                WITH a, b, row
                WHERE a IS NOT NULL AND b IS NOT NULL
                CALL apoc.create.relationship(a, row.rel_type, coalesce(row.properties, {{}}), b) YIELD rel
                RETURN 1
              ",
              {{
                batchSize: {batch_size},
                parallel: true,
                params: {{rows: $rows}}
              }}
            )
        """

        with self.driver.session() as session:
            params = {"rows": relationships}
            result = session.run(cypher, params)
            for record in result:
                # apoc.periodic.iterate 返回的是 total 字段
                if 'total' in record:
                    return record['total']
            return len(relationships)

    def batch_create_relationships_apoc(
            self, head_label, tail_label, relationships, batch_size=1000, data_limit=100_000
    ):
        """
        使用 APOC 批量加速创建节点间动态类型关系，Python 侧分批，显示进度条。

        :param head_label: 起始节点标签
        :param tail_label: 目标节点标签
        :param relationships: [
            {
                "head_id": ...,
                "tail_id": ...,
                "rel_type": ...,
                "properties": ...,
            },
            ...
        ]
        :param batch_size: 每批处理数量
        :param data_limit: 最大允许插入的关系数量
        :return: 实际插入的关系数量
        """
        total = len(relationships)
        if not relationships:
            return 0

        if total > data_limit:
            raise ValueError(f"单次插入关系数量 {total} 超过上限 {data_limit}，请分批处理！")

        cypher = f"""
            CALL apoc.periodic.iterate(
              "UNWIND $rows AS row RETURN row",
              "
                MATCH (a:{head_label} {{id: row.head_id}})
                MATCH (b:{tail_label} {{id: row.tail_id}})
                WITH a, b, row
                WHERE a IS NOT NULL AND b IS NOT NULL
                CALL apoc.create.relationship(a, row.rel_type, coalesce(row.properties, {{}}), b) YIELD rel
                RETURN 1
              ",
              {{
                batchSize: {batch_size},
                parallel: true,
                params: {{rows: $rows}}
              }}
            )
        """

        with self.driver.session() as session:
            params = {"rows": relationships}
            result = session.run(cypher, params)
            for record in result:
                if 'total' in record:
                    return record['total']
            return len(relationships)


    def create_or_update_relationship(self, head_id: str, tail_id: str,
                                      rel_type: str, properties: Dict = None,
                                      head_label: str = "", tail_label: str = "") -> None:
        h_label_str = f":{head_label}" if head_label else ""
        t_label_str = f":{tail_label}" if tail_label else ""

        check_cypher = f"""
        MATCH (a{h_label_str} {{id: $h_id}})
        MATCH (b{t_label_str} {{id: $t_id}})
        RETURN count(a) AS a_count, count(b) AS b_count
        """

        create_cypher = f"""
        MATCH (a{h_label_str} {{id: $h_id}})
        MATCH (b{t_label_str} {{id: $t_id}})
        MERGE (a)-[r:{sanitize_label(rel_type)}]->(b)
        {'SET r += $props' if properties else ''}
        RETURN r
        """

        params = {
            "h_id": head_id,
            "t_id": tail_id,
            "props": properties or {}
        }

        with self.driver.session() as session:
            def check_nodes(tx):
                result = tx.run(check_cypher, **params)
                record = result.single()
                if record["a_count"] == 0:
                    raise ValueError(f"Head node (id={head_id}) does not exist.")
                if record["b_count"] == 0:
                    raise ValueError(f"Tail node (id={tail_id}) does not exist.")

            def create_relationship(tx):
                result = tx.run(create_cypher, **params)
                record = result.single()
                if record is None or record["r"] is None:
                    raise RuntimeError("Failed to create or update relationship.")

            session.execute_write(check_nodes)
            try:
                session.execute_write(create_relationship)
            except Neo4jError as e:
                raise RuntimeError(f"Failed to create or update relationship: {e}")

    def create_id_index(self, label):
        """
        为指定标签的节点的 id 属性创建唯一约束。
        不同标签之间 id 可以重复，同标签内 id 唯一。
        """
        cypher = f"""
        CREATE INDEX {label}_index_id IF NOT EXISTS FOR (n:{label}) ON (n.id)
        """
        with self.driver.session() as session:
            session.run(cypher)


    def clean_database(self):
        """
        为指定标签的节点的 id 属性创建唯一约束。
        不同标签之间 id 可以重复，同标签内 id 唯一。
        """
        cypher = f"""
        match(n) detach delete n
        """
        with self.driver.session() as session:
            session.run(cypher)


    def batch_create_relationships(self, head_label, tail_label, relation_type, relationships, batch_size=1000):
        """
        批量创建节点间关系（不检查节点是否存在）。

        :param head_label: 起始节点标签
        :param tail_label: 目标节点标签
        :param relation_type: 关系类型（字符串）
        :param relationships: [
            {
                "head_id": ...,
                "tail_id": ...,
                "properties": ...,
            },
            ...
        ]
        :param batch_size: 每批次插入的数量，默认1000
        """
        if not relationships:
            return

        head_label = head_label
        tail_label = tail_label
        rel_type = sanitize_label(relation_type)

        def chunked_iterable(iterable, size):
            for i in range(0, len(iterable), size):
                yield iterable[i:i + size]

        cypher = f"""
        UNWIND $rows AS row
        MATCH (a:{head_label} {{id: row.head_id}})
        MATCH (b:{tail_label} {{id: row.tail_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += coalesce(row.properties, {{}})
        """

        with self.driver.session() as session:
            for batch in tqdm(chunked_iterable(relationships, batch_size), desc="Adding relationships by batch"):
                session.execute_write(lambda tx: tx.run(cypher, rows=batch))

    def create_or_merge_node(self, label, node):
        assert node.get('id') or node.get('id').strip() != "", "Node ID must be provided in properties."
        existing_nodes = self.query_node_by_id(node['id'], label=label)
        existing_node = existing_nodes[0] if len(existing_nodes) > 0 else None
        if existing_node:
            insert_node = merge_dicts(node, existing_node)
        else:
            insert_node = node
        if insert_node == existing_node:
            return
        self.create_or_update_node(label=label, properties=insert_node)

    def query_node_by_id(self, node_id: str, label="") -> List[Dict]:
        """根据ID查询节点

        Args:
            node_id: 节点ID

        Returns:
            节点字典列表
        """
        label = f":{label}" if label else ""
        cypher = f"MATCH (n{label}) WHERE n.id = $id RETURN n"

        with self.driver.session() as session:
            result = session.execute_read(
                lambda tx: tx.run(cypher, id=node_id).data()
            )

        return [record['n'] for record in result]

    def query_node_by_name(self, node_name: str, label="") -> List[Dict]:
        """根据name查询节点

        Args:
            node_id: 节点ID

        Returns:
            节点字典列表
        """
        label = f":{label}" if label else ""
        cypher = f"MATCH (n{label}) WHERE n.mention = $name RETURN n"

        with self.driver.session() as session:
            result = session.execute_read(
                lambda tx: tx.run(cypher, name=node_name).data()
            )

        return [record['n'] for record in result]

    def get_connected_edges_and_nodes(self, node_id: str, label: str = "") -> List[Dict]:
        """根据节点ID和标签获取所有相连的边和节点

        Args:
            node_id: 节点ID
            label: 节点标签（可选）

        Returns:
            包含边和连接节点信息的字典列表。每个字典包含：
            - source_node: 源节点信息
            - edge: 边信息（包括属性和类型）
            - target_node: 目标节点信息
        """
        label_part = f":{label}" if label else ""
        cypher = f"""
        MATCH (n{label_part})-[r]-(m)
        WHERE n.id = $id
        RETURN n, labels(n) as n_labels, r, properties(r) as r_props, m, labels(m) as m_labels
        """

        with self.driver.session() as session:
            result = session.execute_read(
                lambda tx: tx.run(cypher, id=node_id).data()
            )

        return [
            {
                'source_node': {
                    **record['n'],
                    'labels': record['n_labels']
                },
                'edge': {
                    **record['r_props'],  # 关系的属性
                    'begin': record['r'][0],
                    'r': record['r'][1],
                    'end': record['r'][2]
                },
                'target_node': {
                    **record['m'],
                    'labels': record['m_labels']
                }
            }
            for record in result
        ]



    def create_relationship_by_id(
            self,
            head_id: str,
            tail_id: str,
            rel_type: str,
            head_label: str = "",
            tail_label: str = "",
            properties: Optional[Dict] = None
    ) -> None:
        """根据节点ID创建或更新关系

        Args:
            head_id: 起始节点ID
            tail_id: 目标节点ID
            rel_type: 关系类型（自动清洗非法字符）
            head_label: 起始节点标签（自动清洗）
            tail_label: 目标节点标签（自动清洗）
            properties: 关系属性字典

        Raises:
            ValueError: 节点ID为空
            Neo4jError: 数据库操作失败
        """
        # 参数校验
        if not head_id or not tail_id:
            raise ValueError("节点ID不能为空")

        try:
            # 标签及关系类型清洗
            sanitized_rel = sanitize_label(rel_type)
            h_label = f":{sanitize_label(head_label)}" if head_label else ""
            t_label = f":{sanitize_label(tail_label)}" if tail_label else ""

            # 构建参数化Cypher
            cypher = f"""
            MERGE (a{h_label} {{id: $head_id}})
            MERGE (b{t_label} {{id: $tail_id}})
            MERGE (a)-[r:{sanitized_rel}]->(b)
            {'SET r += $props' if properties else ''}
            RETURN id(r) as relationship_id
            """

            # 执行事务
            with self.driver.session() as session:
                session.write_transaction(
                    lambda tx: tx.run(
                        cypher,
                        head_id=head_id,
                        tail_id=tail_id,
                        props=properties
                    )
                )

        except Exception as e:
            # 包装原始异常并添加调试信息
            error_info = {
                "head_id": head_id,
                "tail_id": tail_id,
                "rel_type": rel_type,
                "labels": (head_label, tail_label)
            }
            raise Exception(f"创建关系失败: {error_info}") from e

    def is_connected(
            self,
            head_id: str,
            tail_id: str,
            head_label: str = "",
            tail_label: str = "",
            max_path_length: Optional[int] = None
    ) -> bool:
        """判断两个节点之间是否存在路径（连通性）

        Args:
            head_id: 起始节点ID
            tail_id: 目标节点ID
            head_label: 起始节点标签（可选）
            tail_label: 目标节点标签（可选）
            max_path_length: 最大路径长度限制（可选）

        Returns:
            bool: 如果存在路径则返回 True，否则返回 False

        Raises:
            ValueError: 节点ID为空
            Exception: 数据库操作失败
        """
        # 参数校验
        if not head_id or not tail_id:
            raise ValueError("节点ID不能为空")

        try:
            # 标签清洗
            h_label = f":{sanitize_label(head_label)}" if head_label else ""
            t_label = f":{sanitize_label(tail_label)}" if tail_label else ""

            # 构建路径长度限制
            path_length = f"*..{max_path_length}" if max_path_length is not None else "*"

            # 构建参数化Cypher
            cypher = f"""
            MATCH (a{h_label} {{id: $head_id}}), (b{t_label} {{id: $tail_id}})
            RETURN EXISTS((a)-[{path_length}]-(b)) AS connected
            """

            # 执行事务
            with self.driver.session() as session:
                result = session.read_transaction(
                    lambda tx: tx.run(
                        cypher,
                        head_id=head_id,
                        tail_id=tail_id
                    ).single()
                )

                # 返回连通性结果
                return result["connected"]

        except neo4j_exceptions.Neo4jError as e:
            # 包装原始异常并添加调试信息
            error_info = {
                "head_id": head_id,
                "tail_id": tail_id,
                "labels": (head_label, tail_label),
                "max_path_length": max_path_length
            }
            raise Exception(f"连通性查询失败: {error_info}") from e

    def find_shortest_path(
            self,
            head_id: str,
            tail_id: str,
            head_label: str = "",
            tail_label: str = "",
    ) -> Optional[List[Tuple[str, str]]]:
        """查找两个节点之间的最短路径

        Args:
            head_id: 起始节点ID
            tail_id: 目标节点ID
            head_label: 起始节点标签（可选）
            tail_label: 目标节点标签（可选）

        Returns:
            Optional[List[Tuple[str, str]]]: 如果存在路径，则返回由节点对组成的列表表示的路径；如果不存在路径，则返回 None

        Raises:
            ValueError: 节点ID为空
            Exception: 数据库操作失败
        """
        # 参数校验
        if not head_id or not tail_id:
            raise ValueError("节点ID不能为空")

        if head_id == tail_id:
            return [{
                'begin': self.query_node_by_id(head_id, head_label)[0],
                'r': "same_as",
                'end': self.query_node_by_id(tail_id, tail_label)[0]
            }]
        try:
            # 标签清洗
            h_label = f":{sanitize_label(head_label)}" if head_label else ""
            t_label = f":{sanitize_label(tail_label)}" if tail_label else ""

            # 构建参数化Cypher
            cypher = f"""
            MATCH (a{h_label} {{id: $head_id}}), (b{t_label} {{id: $tail_id}}),
            p = shortestPath((a)-[*]-(b))
            RETURN p
            """

            # 执行事务
            results = []
            with self.driver.session() as session:
                result = session.read_transaction(
                    lambda tx: tx.run(
                        cypher,
                        head_id=head_id,
                        tail_id=tail_id
                    ).single(),
                )

                # 如果没有结果，说明没有路径
                if not result or not result["p"]:
                    return None

                # 解析路径
                path = result["p"]
                results = [{'begin': dict(rel.start_node),
                            'r': rel.type,
                            'end': dict(rel.start_node),
                            **dict(rel)
                            } for rel in path.relationships]
                return results

        except Neo4jError as e:
            # 包装原始异常并添加调试信息
            error_info = {
                "head_id": head_id,
                "tail_id": tail_id,
                "labels": (head_label, tail_label)
            }
            raise Exception(f"最短路径查询失败: {error_info}") from e


    def query_cooccurrence_entities(self, entity_id: str) -> List[Dict]:
            """
            查询与指定 Entity 通过 Doc 共现的其他 Entity 及对应 Doc

            Args:
                entity_id: Entity 节点的 id

            Returns:
                包含 'entity' 和 'docs' 的字典列表
            """
            cypher = """
                MATCH (e1:Entity {id: $entity_id})-[]-(doc:Doc)-[]-(e2:Entity)
                WHERE e1 <> e2
                RETURN e2 AS entity, collect(DISTINCT doc) AS docs
            """

            entities = []
            with self.driver.session() as session:
                result = session.execute_read(
                    lambda tx: tx.run(cypher, entity_id=entity_id).data()
                )

                for res in result:
                    res['entity']['source'] = res['docs'][0]['title']
                    entities.append(res['entity'])
            return entities

    def query_cooccurrence_entities_with_docs_degree_limit(
        self,
        entity_id: str,
        domain: Optional[str],
        degree_limit: int = 100,
        timeout_seconds: Optional[float] = 60
    ) -> List[Dict]:
        """
        查询与指定 Entity 通过 Doc 共现的其他 Entity 及对应 Doc，并可设置查询超时。
        （原有的 "过滤掉度数大于200的实体" 逻辑并未在Cypher中体现，请注意）
        """
        if domain:
            domain_str = f"(e1:Entity:{domain} {{id: $entity_id}})-[]-(doc:Doc:{domain})-[]-(e2:Entity:{domain})"
        else:
            domain_str = f"(e1:Entity {{id: $entity_id}})-[]-(doc:Doc)-[]-(e2:Entity)"

        # 注意：原函数名中的 degree_limit 和注释中的 "过滤掉度数大于200的实体"
        # 在您提供的 Cypher 查询中并未使用。如果需要，应在 WHERE 子句中加入类似
        # "WHERE size((e2)--()) < 200" 的逻辑。

        cypher = f"""
               MATCH {domain_str}
               WHERE e1 <> e2
               RETURN e2 AS entity, doc
           """

        doc_ents = []
        with self.driver.session() as session:
            # 2. 准备要传递给 run 方法的参数
            params = {"entity_id": entity_id}

            # 3. 执行查询，并传入超时参数
            # 使用 session.execute_read 保证在只读事务中执行
            result = session.execute_read(
                lambda tx: tx.run(
                    cypher,
                    params,
                    timeout=timeout_seconds # <--- 将超时时间传给 run 方法
                ).data()
            )

            for res in result:
                # 确保 res['entity'] 和 res['doc'] 不是 None
                if res.get('entity') and res.get('doc'):
                    # 最好使用 .get() 方法来安全地访问字典键
                    res['entity']['source'] = res['doc'].get('title', 'N/A')
                    doc_ents.append((res['entity'], res['doc']))

        return doc_ents

    def query_cooccurrence_entities_with_docs(self, entity_id: str, domain) -> List[Dict]:
            """
            查询与指定 Entity 通过 Doc 共现的其他 Entity 及对应 Doc

            Args:
                entity_id: Entity 节点的 id

            Returns:
                包含 'entity' 和 'docs' 的字典列表
            """


            if domain:
                domain_str = f"(e1:Entity:{domain} {{id: $entity_id}})-[]-(doc:Doc:{domain})-[]-(e2:Entity:{domain})"
            else:
                domain_str = f"(e1:Entity {{id: $entity_id}})-[]-(doc:Doc)-[]-(e2:Entity)"

            cypher = f"""
                MATCH {domain_str}
                WHERE e1 <> e2
                RETURN e2 AS entity,  doc
            """

            doc_ents = []
            with self.driver.session() as session:
                result = session.execute_read(
                    lambda tx: tx.run(cypher, entity_id=entity_id).data()
                )

                for res in result:
                    res['entity']['source'] = res['doc']['title']
                    doc_ents.append((res['entity'], res['doc']))
            return doc_ents

