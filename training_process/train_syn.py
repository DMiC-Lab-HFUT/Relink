import json
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict, Any

from path_scorer import PathScorer
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

# 假设以下文件已存在
import train_relation_encoder
import train_path_scorer
import os
from itertools import islice

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



def _load_data_and_create_loaders(path_batch_size: int, quad_batch_size: int):
    """
    加载数据集并创建数据加载器。

    Args:
        path_batch_size: 每个批次的样本数量。


    Returns:
        返回一个包含四个 DataLoader 的元组：(train_quads_loader, test_quads_loader,
                                            train_path_loader, val_path_loader)。
    """
    # 1. 加载和处理 quads 数据
    with open('quads.json', 'r', encoding='utf-8') as f:
        quads_data = json.load(f)
    random.shuffle(quads_data)
    quads_data = random.sample(quads_data, len(quads_data))
    split_quads_idx = int(len(quads_data) * 0.9)
    train_quads = quads_data[:split_quads_idx]
    test_quads = quads_data[split_quads_idx:]

    train_quads_loader = DataLoader(
        train_relation_encoder.QuadDataset(train_quads),
        batch_size=quad_batch_size, shuffle=True,
        collate_fn=train_relation_encoder.collate_fn
    )
    test_quads_loader = DataLoader(
        train_relation_encoder.QuadDataset(test_quads),
        batch_size=quad_batch_size,
        collate_fn=train_relation_encoder.collate_fn
    )

    # 2. 加载和处理 path samples 数据
    path_samples = []
    with open('path_samples.jsonl', encoding='utf-8') as f:
        for line_index, line in enumerate(tqdm(f, desc="加载路径样本")):
            # if line_index > 100000:
            #     break
            path_samples.append(json.loads(line))
    random.shuffle(path_samples)
    split_path_idx = int(len(path_samples) * 0.9)
    train_paths = path_samples[:split_path_idx]
    val_paths = path_samples[split_path_idx:]

    train_path_loader = DataLoader(
        train_path_scorer.PathDataset(train_paths),
        batch_size=path_batch_size, shuffle=True,
        collate_fn=train_path_scorer.collate_fn
    )
    val_path_loader = DataLoader(
        train_path_scorer.PathDataset(val_paths),
        batch_size=path_batch_size, shuffle=True,
        collate_fn=train_path_scorer.collate_fn
    )

    return train_quads_loader, test_quads_loader, train_path_loader, val_path_loader


def train_ranker_one_epoch(scorer, accelerator, optimizer, train_loader, val_loader, epoch):
    """
    执行一个周期的排名（Ranker）任务训练和验证。

    Args:
        scorer: PathScorer 模型。
        accelerator: Accelerate 实例。
        optimizer: 优化器。
        train_loader: 训练数据加载器。
        val_loader: 验证数据加载器。
        epoch: 当前的训练周期数。

    Returns:
        当前周期的验证损失。
    """
    criterion = torch.nn.MSELoss()
    pairwise_loss_fn = train_path_scorer.make_pairwise_loss('logistic')

    train_stats = train_path_scorer.run_epoch(
        epoch, scorer, train_loader, torch.nn.MSELoss(), pairwise_loss_fn, optimizer, accelerator, train=True
    )
    val_stats = train_path_scorer.run_epoch(
        epoch, scorer, val_loader, criterion, pairwise_loss_fn, None, accelerator, train=False
    )
    accelerator.wait_for_everyone()

    train_loss, train_mse, train_mae, train_pearson, train_rank_loss, train_rank_pair_count, train_rank_acc = train_stats
    val_loss, val_mse, val_mae, val_pearson, val_rank_loss, val_rank_pair_count, val_rank_acc = val_stats

    if accelerator.is_main_process:
        print(f"\n--- Epoch {epoch} Ranker Summary ---")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(
            f"  Train Metrics: MSE={train_mse:.4f}, MAE={train_mae:.4f}, Pearson={train_pearson:.4f}, RankAcc={train_rank_acc:.4f}"
        )
        print(
            f"  Val Metrics:   MSE={val_mse:.4f}, MAE={val_mae:.4f}, Pearson={val_pearson:.4f}, RankAcc={val_rank_acc:.4f}"
        )
        print(f"  Train RankLoss: {train_rank_loss:.4f} (on {train_rank_pair_count} pairs)")
        print(f"  Val RankLoss:   {val_rank_loss:.4f} (on {val_rank_pair_count} pairs)\n")

    return val_loss


def align_one_epoch(scorer, accelerator, optimizer, train_loader, val_loader, epoch):
    """
    执行一个周期的对齐（Aligner）任务训练和验证。

    Args:
        scorer: PathScorer 模型。
        accelerator: Accelerate 实例。
        optimizer: 优化器。
        train_loader: 训练数据加载器。
        val_loader: 验证数据加载器。
        epoch: 当前的训练周期数。

    Returns:
        当前周期的验证损失。
    """
    scorer.train()
    unwrapped_scorer = accelerator.unwrap_model(scorer)
    train_loss, _, _ = train_relation_encoder.run_epoch(
        accelerator=accelerator,
        doc_relation_encoder=unwrapped_scorer.doc_rel_encoder,
        relation_encoder=unwrapped_scorer.triple_encoder,
        loader=train_loader,
        optimizer=optimizer,
        train=True
    )

    scorer.eval()
    val_loss, _, _ = train_relation_encoder.run_epoch(
        accelerator=accelerator,
        doc_relation_encoder=unwrapped_scorer.doc_rel_encoder,
        relation_encoder=unwrapped_scorer.triple_encoder,
        loader=val_loader,
        optimizer=optimizer,
        train=False
    )
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"--- Epoch {epoch} Aligner Summary ---")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n")

    return val_loss


def main(early_stop=3, num_epochs=20, batch_size=16):
    """
    主训练函数，负责模型的端到端训练流程。

    Args:
        early_stop: 验证损失连续没有改善的最大周期数，超过则提前终止训练。
        num_epochs: 总共训练的周期数。
        batch_size: 每个批次的样本数量。
        max_path_samples: 从 path_samples_.jsonl 中读取的最大样本数。
    """
    # 1. 初始化 Accelerate 库
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[ddp_kwargs])

    # 2. 加载数据并创建数据加载器
    train_quads_loader, test_quads_loader, train_path_loader, val_path_loader = \
        _load_data_and_create_loaders(quad_batch_size=30, path_batch_size=100)

    # 3. 初始化模型和优化器
    scorer = PathScorer()
    # scorer.load_encoder('./checkpoints/encoder.pth')
    scorer_dict = torch.load("checkpoints/scorer_best.pth")
    scorer.load_state_dict(scorer_dict)
    optimizer = torch.optim.Adam(
        scorer.parameters(),
        lr=2e-5,
        weight_decay=5e-4
    )

    # 4. 包装所有组件以进行分布式训练
    scorer, optimizer, train_quads_loader, test_quads_loader, train_path_loader, val_path_loader = accelerator.prepare(
        scorer, optimizer, train_quads_loader, test_quads_loader, train_path_loader, val_path_loader
    )

    # 5. 开始训练循环
    best_combined_loss, patience = float('inf'), 0
    for epoch in range(num_epochs):

        # 训练和评估排名任务
        val_loss_ranker = train_ranker_one_epoch(scorer, accelerator, optimizer, train_path_loader, val_path_loader, epoch)

        # 训练和评估对齐任务
        val_loss_align = align_one_epoch(scorer, accelerator, optimizer, train_quads_loader, test_quads_loader, epoch)

        # 使用两个任务的验证损失之和作为早停标准
        combined_val_loss = val_loss_ranker + val_loss_align

        if accelerator.is_main_process:
            if combined_val_loss < best_combined_loss:
                best_combined_loss, patience = combined_val_loss, 0
                unwrapped_scorer = accelerator.unwrap_model(scorer)
                accelerator.save(unwrapped_scorer.state_dict(), './checkpoints/scorer_best.pth')
                print(f"✅ 模型在 epoch {epoch} 得到改善并已保存，组合验证损失：{combined_val_loss:.4f}")
            else:
                patience += 1
                if patience >= early_stop:
                    print(f"早停条件在 epoch {epoch} 触发，训练终止。")
                    break


if __name__ == "__main__":
    # 在这里可以轻松调整训练参数
    main(num_epochs=30, early_stop=3, batch_size=16)



