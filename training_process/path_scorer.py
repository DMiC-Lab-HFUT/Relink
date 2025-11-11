import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Any
import os
os.environ["TOKENIZERS_PARALLELISM"] = "True"

def freeze_bert_layers(model, unfreeze_last_n=3):
    """
    冻结 BERT 的前层，只微调最后 n 层和 pooler。
    model: AutoModel (如 from_pretrained)
    unfreeze_last_n: 保留可训练的最后 n 层
    """
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 解冻最后 n 层
    for i in range(12 - unfreeze_last_n, 12):
        for param in model.encoder.layer[i].parameters():
            param.requires_grad = True

    # 如果你需要微调 pooler
    if hasattr(model, 'pooler'):
        for param in model.pooler.parameters():
            param.requires_grad = True


def freeze_all_but_layer(model, layer_names=[]):
    """
    冻结除指定层以外的所有参数
    layer_names: 需要训练的层的名字列表
    """
    for name, param in model.named_parameters():
        if any([name.startswith(layer_name) for layer_name in layer_names]):
            param.requires_grad = True
        else:
            param.requires_grad = False


def is_sublist(a, b):
    """
    判断列表a是否为b的子串（连续子序列），
    如果是，返回起始位置，否则返回None
    """
    n, m = len(a), len(b)
    for i in range(m - n + 1):
        if b[i:i + n] == a:
            return i
    return None


class DocRelationEncoder(nn.Module):
    def __init__(self, pretrained_model='./bert-base-uncased', encoder = None, window_size=512):
        super().__init__()
        if encoder is None:
            self.bert = AutoModel.from_pretrained(pretrained_model)
            freeze_bert_layers(self.bert, 6)
        else:
            self.bert = encoder
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=False)

        # self.relation_proj = nn.Linear(768, 768)
        self.window_size = window_size
        # self.device = device #
        # self.to(device)      #

    def forward(self, h, t, doc, rel_only=False):
        assert len(h) == len(t) == len(doc), "Input lists must be the same length"

        templates = [f' | The relationship between "{hh}" and "{tt}" is [MASK]' for hh, tt in zip(h, t)]
        template_token_lens = [len(self.tokenizer.tokenize(temp)) for temp in templates]

        window_texts = [
            self.extract_window(
                self.tokenizer, d, self.window_size - 2 - tmpl_len, hh, tt
            )
            for d, hh, tt, tmpl_len in zip(doc, h, t, template_token_lens)
        ]

        processed_texts = [text + template for text, template in zip(window_texts, templates)]
        inputs = self.tokenizer(processed_texts, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(self.bert.device) for k, v in inputs.items()}

        outputs = self.bert(**inputs).last_hidden_state

        mask_id = self.tokenizer.convert_tokens_to_ids('[MASK]')
        mask_emb = self.extract_mask_emb((inputs['input_ids'] == mask_id), outputs)
        rel_emb = mask_emb  # self.relation_proj(mask_emb)

        if rel_only:
            return rel_emb

        h_emb = self._encode_entity(h)
        t_emb = self._encode_entity(t)
        output = torch.cat([h_emb, rel_emb, t_emb], dim=-1)
        return output

    def _encode_entity(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(self.bert.device) for k, v in inputs.items()}
        return self.bert(**inputs).last_hidden_state[:, 0]

    def extract_window(self, tokenizer, text, window_size, e1, e2):
        tokens = tokenizer.tokenize(text)
        e1_tokens = tokenizer.tokenize(e1)
        e2_tokens = tokenizer.tokenize(e2)
        container = set()
        e1_start = is_sublist(e1_tokens, tokens)
        e2_start = is_sublist(e2_tokens, tokens)
        if e1_start is None or e2_start is None:
            return tokenizer.convert_tokens_to_string(tokens)[:window_size]

        e1_end = e1_start + len(e1_tokens)
        e2_end = e2_start + len(e2_tokens)
        container.update(list(range(e1_start, e1_end + 1)))
        container.update(list(range(e2_start, e2_end + 1)))

        left_border = 0
        right_border = len(tokens) - 1
        while len(container) < len(tokens) and len(container) < window_size:
            e1_start -= 1
            e1_end += 1
            e2_start -= 1
            e2_end += 1
            container.update([index for index in [e1_start, e2_start, e1_end, e2_end] if
                              index >= left_border and index <= right_border])

        indices = [x for x in sorted(container) if x >= left_border and x <= right_border]
        window_tokens = [tokens[i] for i in indices]
        window_text = tokenizer.convert_tokens_to_string(window_tokens)
        return window_text

    def extract_mask_emb(self, mask, outputs):
        idx = mask.int().argmax(dim=1)
        emb = outputs[torch.arange(outputs.size(0)), idx]
        return emb


class TripleEncoder(nn.Module):
    def __init__(self, pretrained_model='./bert-base-uncased', encoder = None):
        super().__init__()
        if encoder is None:
            self.bert = AutoModel.from_pretrained(pretrained_model)
            freeze_bert_layers(self.bert, 6)
        else:
            self.bert = encoder
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=False)
        # self.relation_proj = nn.Linear(768, 768)

    def forward(self, h: List[str], t: List[str], r: List[str], rel_only: bool = False) -> torch.Tensor:
        """
        修正后的forward方法，将h,r,t合并进行一次性编码，避免模型重入导致的inplace error。
        如果 rel_only=True，则只编码并返回关系 r 的 embedding。
        """
        # 首先对关系r的文本进行预处理，因为无论哪种模式都可能用到它
        r = [_r.replace("_", " ") for _r in r]

        # ✅ 关键改动：根据 rel_only 的值进行判断
        if rel_only:
            # --- 如果只编码关系 r ---
            batch_size = len(r)
            if batch_size == 0:
                # 返回一个形状正确的空张量，维度是单个embedding的维度
                return torch.empty(0, 768, device=self.bert.device)

            # 直接调用编码函数，只编码关系 r
            r_emb = self._encode_entity(r)
            return r_emb
        else:
            # --- 如果编码并拼接 h, r, t (之前的逻辑) ---
            batch_size = len(h)
            if batch_size == 0:
                # 返回一个形状正确的空张量，维度是三者拼接后的维度
                return torch.empty(0, 768 * 3, device=self.bert.device)

            # 1. 将 h, r, t 的文本列表合并成一个大的列表
            all_texts = h + r + t

            # 2. 对这个大列表只进行一次编码
            # all_embs 的形状是 [batch_size * 3, hidden_size]
            all_embs = self._encode_entity(all_texts)

            # 3. 将编码结果切分并重组成 [h, r, t] 的形式
            # embs_chunked 的形状是 [3, batch_size, hidden_size]
            embs_chunked = all_embs.view(3, batch_size, -1)

            h_emb = embs_chunked[0]  # Shape: [batch_size, hidden_size]
            r_emb = embs_chunked[1]
            t_emb = embs_chunked[2]

            # 4. 最终拼接成 [h_emb, r_emb, t_emb] 的形式
            return torch.cat([h_emb, r_emb, t_emb], dim=-1)


    # def forward(self, h, t, r, rel_only=False):
    #     try:
    #         r = [_r.replace("_", " ") for _r in r]
    #         r_inputs = self.tokenizer(r, padding=True, truncation=True, return_tensors='pt')
    #     except Exception as e:
    #         print(e)
    #     r_inputs = {k: v.to(self.bert.device) for k, v in r_inputs.items()}
    #     r_outputs = self.bert(**r_inputs).last_hidden_state[:, 0]
    #     r_emb = r_outputs
    #
    #     if rel_only:
    #         return r_emb
    #
    #     h_emb = self._encode_entity(h)
    #     t_emb = self._encode_entity(t)
    #     return torch.cat([h_emb, r_emb, t_emb], dim=-1)
    def _encode_entity(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(self.bert.device) for k, v in inputs.items()}
        return self.bert(**inputs).last_hidden_state[:, 0]


class PathAggregator(nn.Module):
    # def __init__(self, input_dim, hidden_dim=512):
    #     super().__init__()
    #     self.gru = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)
    #     self.attention = nn.Sequential(
    #         nn.Linear(2 * hidden_dim, 128),
    #         nn.Tanh(),
    #         nn.Linear(128, 1)
    #     )
    def __init__(self, input_dim, hidden_dim=768, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim,
                          num_layers=num_layers,
                          bidirectional=True,
                          batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.layernorm = nn.LayerNorm(2 * hidden_dim)

    def forward(self, triples_emb, mask=None):
        # 用注意力聚合能让模型自动聚焦于序列中最重要的信息，得到更全面、更有效的路径/序列嵌入，而不仅仅依赖于最后一个时间步的状态。
        # triples_emb: [batch, seq_len, input_dim]
        outputs, _ = self.gru(triples_emb)  # [batch, seq_len, 2*hidden_dim]
        outputs = self.layernorm(outputs)
        attn_logits = self.attention(outputs).squeeze(-1)  # [batch, seq_len]

        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_logits, dim=1)  # [batch, seq_len]
        attn_weights = attn_weights.unsqueeze(-1)  # [batch, seq_len, 1]
        agg = torch.sum(attn_weights * outputs, dim=1)  # [batch, 2*hidden_dim]
        return agg


class QueryInteraction(nn.Module):
    def __init__(self, path_dim, query_dim):
        super().__init__()
        self.path_proj = nn.Linear(path_dim, 256)
        self.query_proj = nn.Linear(query_dim, 256)
        self.mlp = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, path_emb, query_emb):
        path_proj = F.gelu(self.path_proj(path_emb))
        query_proj = F.gelu(self.query_proj(query_emb))
        combined = torch.cat([path_proj, query_proj], dim=-1)
        return self.mlp(combined)


class CrossAttentionInteraction(nn.Module):
    """
    简化后的交互模块，不再对单 token 序列使用 MultiheadAttention。
    直接将 path_emb 与 query_emb 分别投影到同一隐藏维度，然后拼接，
    最后通过 MLP 输出一个打分值。

    输入：
      - path_emb:  [batch, path_dim]
      - query_emb: [batch, query_dim]
    输出：
      - score:     [batch, 1]
    """

    def __init__(self, path_dim: int, query_dim: int, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        # 将 path_emb 映射到 hidden_dim
        self.path_proj = nn.Linear(path_dim, hidden_dim)
        # 将 query_emb 映射到 hidden_dim
        self.query_proj = nn.Linear(query_dim, hidden_dim)

        # MLP 部分，输入维度是 hidden_dim*2（拼接之后），输出 1 维得分
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, path_emb: torch.Tensor, query_emb: torch.Tensor) -> torch.Tensor:
        """
        path_emb:  [batch, path_dim]
        query_emb: [batch, query_dim]
        return:    [batch, 1]
        """
        # 先对两者做线性投影并加非线性
        path_h = F.relu(self.path_proj(path_emb))  # [batch, hidden_dim]
        query_h = F.relu(self.query_proj(query_emb))  # [batch, hidden_dim]

        # 拼接后过 MLP，得到最终分数
        combined = torch.cat([path_h, query_h], dim=-1)  # [batch, hidden_dim*2]
        score = self.mlp(combined)  # [batch, 1]
        return score


class PathScorer(nn.Module):
    """
    用于根据三元组路径和查询文本进行相关性评分的模型。
    支持实体-关系三元组与共现三元组的不同编码方式，并聚合路径信息。
    """

    def __init__(self, pretrained_model: str = './bert-base-uncased'):
        """
        Args:
            pretrained_model: 预训练BERT模型路径或名称
            device: 运行设备
        """
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        freeze_bert_layers(self.encoder, 4)
        # 三元组编码器（实体-关系-实体）
        self.triple_encoder = TripleEncoder(encoder=self.encoder)

        # 文档共现关系编码器
        self.doc_rel_encoder = DocRelationEncoder(encoder=self.encoder)
        # self.doc_rel_encoder.load_state_dict(torch.load('./checkpoints/doc_relation_encoder.pth'))
        # freeze_all_but_layer(self.doc_rel_encoder)

        # 路径聚合器（聚合三元组序列）
        self.path_aggregator = PathAggregator(input_dim=768*2+768)

        # 查询编码器
        self.query_encoder = AutoModel.from_pretrained(pretrained_model)
        # freeze_all_but_layer(self.query_encoder)
        freeze_bert_layers(self.query_encoder, 3)


        # 路径与查询交互层
        self.interaction = CrossAttentionInteraction(path_dim=768*2, query_dim=768)

        # 路径、三元组与查询交互层（可与interaction共用，也可独立）
        self.triple_interaction = CrossAttentionInteraction(
            path_dim=768*2 + (768*2+768),
            query_dim=768
        )

        # 可学习的空路径embedding（用于路径长度为1的情况）
        self.empty_path_emb = nn.Parameter(torch.zeros(768*2))
        nn.init.normal_(self.empty_path_emb, mean=0.0, std=0.02)


    def load_encoder(self, path, unfree_layers=4):
        encoder_dict = torch.load(path)
        self.encoder.load_state_dict(encoder_dict)
        freeze_bert_layers(self.encoder, unfree_layers)

    def triple_classify(self, padded_paths):
        """
        将三元组区分为实体-关系型和共现型，便于分别编码。
        Args:
            padded_paths: List[List[Tuple]]，每条路径已pad到统一长度
        Returns:
            (real_triples, cooccur_triples)
        """
        i_real, h_real, r_real, t_real, docs_real = [], [], [], [], []
        i_cooccur, h_cooccur, r_cooccur, t_cooccur, docs_cooccur = [], [], [], [], []

        triples = [triple for path in padded_paths for triple in path]
        for i, (h, r, t, doc) in enumerate(triples):
            if r == 'co-occurrence':
                i_cooccur.append(i)
                h_cooccur.append(h)
                t_cooccur.append(t)
                r_cooccur.append(r)
                docs_cooccur.append(doc)
            else:
                i_real.append(i)
                h_real.append(h)
                r_real.append(r)
                t_real.append(t)
                docs_real.append(doc)
        return (i_real, h_real, t_real, r_real, docs_real), (i_cooccur, h_cooccur, t_cooccur, r_cooccur, docs_cooccur)

    def pad_paths(self, triples_list_batch):
        """
        对batch内所有路径进行padding，使其长度一致。
        Returns:
            padded_paths: List[List[Tuple]]
            path_lens: List[int]
        """
        batch_size = len(triples_list_batch)
        max_path_len = max(len(path) for path in triples_list_batch)
        pad_triple = ("[PAD]", "[PAD]", "[PAD]", "[PAD]")
        padded_paths, path_lens = [], []
        for path in triples_list_batch:
            path_lens.append(len(path))
            padded_path = path + [pad_triple] * (max_path_len - len(path))
            padded_paths.append(padded_path)
        return padded_paths, path_lens, max_path_len

    def forward(self, triples_list_batch, query_texts, return_full_path_score=False):
        """
        Args:
            triples_list_batch: List[List[Tuple]]，每条路径由三元组组成
            query_texts: List[str]，查询文本
            return_full_path_score: (可选) 是否同时返回完整路径的得分，默认为False
        Returns:
            last_triple_scores: [batch, 1] (默认)
            (path_scores, last_triple_scores): (可选)
        """
        # 1. 填充路径并分类
        padded_paths, path_lens, max_path_len = self.pad_paths(triples_list_batch)
        (i_real, h_real, t_real, r_real, docs_real), \
            (i_cooccur, h_cooccur, t_cooccur, r_cooccur, docs_cooccur) = self.triple_classify(padded_paths)

        # 2. 编码所有三元组
        embeddings = []
        if len(i_real) > 0:
            real_triple_embs = self.triple_encoder(h_real, t_real, r_real)
            embeddings += [(i, emb) for (i, emb) in zip(i_real, real_triple_embs)]
        if len(i_cooccur) > 0:
            doc_triple_embs = self.doc_rel_encoder(h_cooccur, t_cooccur, docs_cooccur)
            embeddings += [(i, emb) for (i, emb) in zip(i_cooccur, doc_triple_embs)]
        # 3. 重组为批次形式的 embedding
        embeddings_sorted = sorted(embeddings, key=lambda x: x[0])
        emb_list = [emb for i, emb in embeddings_sorted]
        triple_embs = torch.stack(emb_list, dim=0)
        triple_embs = triple_embs.view(len(triples_list_batch), max_path_len, -1)
        # 4. 创建 mask 用于路径聚合
        mask = torch.zeros(len(triples_list_batch), max_path_len, dtype=torch.float,
                           device=self.triple_encoder.bert.device)
        for i, plen in enumerate(path_lens):
            mask[i, :plen] = 1

        # 5. 编码查询
        query_inputs = self.tokenizer(query_texts, padding=True, truncation=True, return_tensors='pt')
        query_inputs = {k: v.to(self.triple_encoder.bert.device) for k, v in query_inputs.items()}
        query_emb = self.query_encoder(**query_inputs).last_hidden_state[:, 0]

        # --- 默认计算 last_triple_scores ---

        # 6. 计算排除最后一个三元组的路径 embedding (向量化实现)
        mask_excl = mask.clone()
        long_path_indices = torch.tensor(path_lens, device=mask.device) > 1
        if long_path_indices.any():
            rows_to_update = torch.where(long_path_indices)[0]
            last_element_indices = (torch.tensor(path_lens, device=mask.device)[long_path_indices] - 1)
            mask_excl[rows_to_update, last_element_indices] = 0

        batch_path_emb_excl = self.path_aggregator(triple_embs, mask_excl)
        path_emb_excl = torch.where(
            long_path_indices.view(-1, 1),
            batch_path_emb_excl,
            self.empty_path_emb.to(query_emb.device)
        )

        # 7. 获取最后一个三元组的 embedding
        last_triple_embs = torch.stack(
            [triple_embs[i, plen - 1] for i, plen in enumerate(path_lens)], dim=0
        )

        # 8. 拼接并计算默认要返回的 last_triple_scores
        concat_emb = torch.cat([path_emb_excl, last_triple_embs], dim=1)
        last_triple_scores = self.triple_interaction(concat_emb, query_emb)

        # --- 新的返回逻辑 ---
        if return_full_path_score:
            # 仅在需要时才计算完整路径的得分
            path_emb = self.path_aggregator(triple_embs, mask)
            path_scores = self.interaction(path_emb, query_emb)
            return path_scores, last_triple_scores
        else:
            # 默认只返回 last_triple_scores
            return last_triple_scores
# ==== 测试代码 ====
if __name__ == '__main__':
    triples_list_batch = [
        [  # 长度2
            ("Barack Obama", "born_in", "Hawaii", "doc_1"),
            ("Hawaii", "located_in", "USA", "doc_1"),
        ],
        [  # 长度3
            ("Beijing", "capital_of", "China", "doc_1"),
            ("China", "located_in", "Asia", "doc_1"),
            ("Asia", "part_of", "Earth", "doc_1"),
        ],
        [  # 长度4
            ("Paris", "located_in", "France", "doc_1"),
            ("France", "part_of", "Europe", "doc_1"),
            ("Europe", "on_continent", "Eurasia", "doc_1"),
            ("Eurasia", "co-occurrence", "5B",
             "Eurasia, the largest continental landmass on Earth, is home to a population of approximately 5B people. This vast region, which combines both Europe and Asia, encompasses a diverse range of cultures, languages, and ethnic groups."),
        ],
        [  # 长度5
            ("Alan Turing", "born_in", "London", "doc_1"),
            ("London", "located_in", "UK", "doc_1"),
            ("UK", "part_of", "Europe", "doc_1"),
            ("Europe", "on_continent", "Eurasia", "doc_1"),
            ("Eurasia", "co-occurrence", "5B",
             "Eurasia, the largest continental landmass on Earth, is home to a population of approximately 5B people. This vast region, which combines both Europe and Asia, encompasses a diverse range of cultures, languages, and ethnic groups."),
        ],
    ]

    query_texts = [
        "Where was Obama born?",
        "What is the capital of China?",
        "Which continent is Paris located in?",
        "Where was Alan Turing born?",
    ]
    model = PathScorer()
    model.eval()
    with torch.no_grad():
        last_triple_scores = model(triples_list_batch, query_texts, return_full_path_score=True)
        print("Last triple scores:", last_triple_scores)  # [batch, 1]
