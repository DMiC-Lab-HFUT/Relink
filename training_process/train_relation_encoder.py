import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Dict, Any

from path_scorer import TripleEncoder, DocRelationEncoder, PathScorer
from torch import nn

# 关键改动：导入 Accelerator
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from transformers import AutoTokenizer, AutoModel

# -----------------------------
# 辅助函数：打印显存
# -----------------------------
def print_memory_usage(accelerator, message=""):
    # 关键的判断：只有主进程才会执行下面的代码
    if accelerator.is_main_process:
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"--- [显存状态: {message}] ---") # 使用普通的 print
        print(f"    已分配 (Allocated): {allocated:.2f} MB")
        print(f"    已预留 (Reserved):  {reserved:.2f} MB")
        print("-----------------------------------")


# -----------------------------
# Utility
# -----------------------------
def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 在使用 accelerate 时，通常不需要设置 cudnn
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


set_seed(42)


# Dataset 和 collate_fn 保持不变
# -----------------------------
# Dataset
# -----------------------------
class QuadDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            'subject': item['subject']['mention'],
            'relation': item['relation'],
            'object': item['object']['mention'],
            'doc': item['doc']
        }


def collate_fn(batch):
    """Combine batch into tensors."""
    return {
        'subjects': [item['subject'] for item in batch],
        'relations': [item['relation'] for item in batch],
        'objects': [item['object'] for item in batch],
        'docs': [item['doc'] for item in batch]
    }


# 损失函数保持不变
# -----------------------------
# Loss Functions
# -----------------------------
def info_nce_loss(triple_emb, doc_emb, temperature=0.05):
    """InfoNCE contrastive loss."""
    triple_emb = F.normalize(triple_emb, dim=-1)
    doc_emb = F.normalize(doc_emb, dim=-1)
    logits = torch.matmul(triple_emb, doc_emb.T) / temperature
    labels = torch.arange(triple_emb.size(0)).to(triple_emb.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def batch_triplet_loss(anchor, positive, negatives, margin=0.5, neg_k=4):
    """
    Triplet loss using hardest of randomly sampled negatives.
    """
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    pos_dist = F.pairwise_distance(anchor, positive)
    B, N, D = negatives.size()

    if N > neg_k:
        idx = torch.randint(0, N, (B, neg_k), device=anchor.device)
        negatives = torch.gather(negatives, 1, idx.unsqueeze(-1).expand(-1, -1, D))

    neg_dist = torch.norm(anchor.unsqueeze(1) - negatives, dim=-1)
    hardest_neg_dist, _ = neg_dist.min(dim=1)
    return F.relu(pos_dist - hardest_neg_dist + margin).mean()


# -----------------------------
# Training Epoch (修改，增加了显存打印)
# -----------------------------
def run_epoch(accelerator, doc_relation_encoder, relation_encoder, loader, optimizer, train=True, neg_k=16):
    if train:
        doc_relation_encoder.train()
        relation_encoder.train()
    else:
        doc_relation_encoder.eval()
        relation_encoder.eval()

    total_loss, total_info_nce, total_triplet = 0, 0, 0
    pbar = tqdm(loader, desc="Training alignment" if train else "Evaluating alignment", disable=not accelerator.is_main_process)

    with torch.set_grad_enabled(train):
        for batch_idx, batch in enumerate(pbar):

            # 前向传播
            doc_emb = doc_relation_encoder(batch['subjects'], batch['objects'], batch['docs'], rel_only=True)


            # with torch.no_grad():
            triple_emb = relation_encoder(batch['subjects'], batch['objects'], batch['relations'], rel_only=True)

            batch_size = doc_emb.size(0)

            # --- 计算损失 ---
            negatives = []
            for i in range(batch_size):
                pool = torch.cat([doc_emb[:i], doc_emb[i + 1:]], dim=0)
                if pool.size(0) > neg_k:
                    idx = torch.randperm(pool.size(0))[:neg_k]
                    negs = pool[idx].unsqueeze(0)
                else:
                    negs = pool.unsqueeze(0)
                negatives.append(negs)
            negatives = torch.cat(negatives, dim=0)

            loss_triplet = batch_triplet_loss(triple_emb, doc_emb, negatives)
            loss_info_nce = info_nce_loss(triple_emb, doc_emb)
            total_loss_batch = 0.00001 * loss_triplet + loss_info_nce

            # --- 训练步骤 ---
            if train:
                optimizer.zero_grad()
                accelerator.backward(total_loss_batch)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(doc_relation_encoder.parameters(), 5.0)
                optimizer.step()


            # --- 累加损失 ---
            gathered_loss = accelerator.gather(total_loss_batch).mean()
            gathered_info_nce = accelerator.gather(loss_info_nce).mean()
            gathered_triplet = accelerator.gather(loss_triplet).mean()

            total_loss += gathered_loss.item()
            total_info_nce += gathered_info_nce.item()
            total_triplet += gathered_triplet.item()

            if accelerator.is_main_process:
                pbar.set_postfix({
                    "loss": f"{gathered_loss.item():.4f}",
                    "info_nce": f"{gathered_info_nce.item():.4f}",
                    "triplet": f"{gathered_triplet.item():.4f}"
                })

    num_batches = len(loader)
    avg_loss = total_loss / num_batches
    avg_info_nce = total_info_nce / num_batches
    avg_triplet = total_triplet / num_batches

    return avg_loss, avg_info_nce, avg_triplet

# -----------------------------
# Training Pipeline
# -----------------------------
def train_model(train_data, val_data, num_epochs=50, batch_size=64, early_stop=3):
    """
    完整的模型训练和验证流程。
    """
    # 1. 初始化 Accelerator
    # find_unused_parameters=True 在模型有部分参数不参与反向传播时（如此处的 triple_encoder）是必需的
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    train_dataset = QuadDataset(train_data)
    val_dataset = QuadDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)




    scorer = PathScorer()
    optimizer = torch.optim.Adam(
        scorer.parameters(),
        lr=2e-5,
        weight_decay=5e-4
    )


    scorer, optimizer, train_loader, val_loader = accelerator.prepare(
        scorer, optimizer, train_loader, val_loader
    )


    # 7. 开始训练循环
    best_val_loss = float('inf')
    patience = 0

    for epoch in range(1, num_epochs + 1):
        if accelerator.is_main_process:
            print(f"\n{'=' * 20} Epoch {epoch}/{num_epochs} {'=' * 20}")

        # --- 训练 ---
        scorer.train()  # 确保训练时模型处于训练模式
        train_loss, _, _ = run_epoch(
            accelerator=accelerator,
            doc_relation_encoder=scorer.doc_rel_encoder,
            relation_encoder=scorer.triple_encoder,
            loader=train_loader,
            optimizer=optimizer,
            train=True
        )

        # --- 验证 ---
        scorer.eval()  # 验证时切换到评估模式
        val_loss, _, _ = run_epoch(
            accelerator=accelerator,
            doc_relation_encoder=scorer.doc_rel_encoder,
            relation_encoder=scorer.triple_encoder,
            loader=val_loader,
            optimizer=optimizer,
            train=False
        )

        # 使用 accelerator.wait_for_everyone() 来同步所有进程
        accelerator.wait_for_everyone()

        # 在主进程上进行日志记录、早停判断和模型保存
        if accelerator.is_main_process:
            print(f"Epoch {epoch} Summary | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                accelerator.print(f"*** 发现新的最优模型，验证损失: {best_val_loss:.4f}。正在保存... ***")
                # 使用 accelerator.unwrap_model 来获取原始模型（去除DDP包装）
                unwrapped_scorer = accelerator.unwrap_model(scorer)

                # 使用 accelerator.save 来确保只在主进程上保存
                accelerator.save(unwrapped_scorer.state_dict(), "./checkpoints/scorer.pth")
            else:
                patience += 1
                accelerator.print(f"验证损失没有提升。Patience: {patience}/{early_stop}")
                if patience >= early_stop:
                    accelerator.print("早停条件触发，训练终止。")
                    break

    accelerator.print("训练流程结束。")



# -----------------------------
# Entry Point
# -----------------------------
if __name__ == '__main__':
    with open('quads.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    random.shuffle(data)
    random.shuffle(data)
    split = int(len(data) * 0.9)
    train_model(data[:split], data[split:])