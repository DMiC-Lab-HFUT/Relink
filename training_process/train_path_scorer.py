import json
import os

import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from tqdm import tqdm
from typing import List, Dict, Any
from accelerate import Accelerator
from path_scorer import PathScorer
from accelerate.utils import DistributedDataParallelKwargs


import argparse

parser = argparse.ArgumentParser(description='训练排序模型...')
parser.add_argument('--exp_name', default="unfree-4", type=str, help='实验名称')
parser.add_argument('--unfree_layers', default=4, type=int, help='共享编码器未冻结层数')
args = parser.parse_args()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def deduplicates_list(lst: List[Any]) -> List[Any]:
    unique_dict = {}
    for item in lst:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key not in unique_dict:
            unique_dict[key] = item
    return list(unique_dict.values())

class PathDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'query': item['query'],
            'path': item['path'],
            'path_score': torch.tensor(item['path_score'], dtype=torch.float),
            'last_triple_score': torch.tensor(item['last_triple_score'], dtype=torch.float)
        }

def collate_fn(batch):
    return {
        'query_texts': [item['query'] for item in batch],
        'triples_list_batch': [item['path'] for item in batch],
        'path_scores': torch.stack([item['path_score'] for item in batch]),
        'last_triple_scores': torch.stack([item['last_triple_score'] for item in batch])
    }

def local_pairwise_accuracy_torch(preds: torch.Tensor, targets: torch.Tensor, epsilon: float = 1.0, min_diff: float = 0.1, batch_size: int = 2000) -> float:
    """
    计算局部成对排序准确率。
    衡量模型在预测具有相似真实分数（差异在 epsilon 内）的样本对的相对顺序方面的能力。
    """
    N = preds.shape[0]
    if N < 2:
        return 1.0  # 如果样本不足以构成一对，则准确率为100%

    device = preds.device
    correct, total = 0, 0

    # 为了节省内存，分批次计算
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_preds = preds[start:end]
        batch_targets = targets[start:end]

        # 使用广播创建所有样本对的分数差异矩阵
        diff_targets = batch_targets.unsqueeze(1) - batch_targets.unsqueeze(0)
        diff_preds = batch_preds.unsqueeze(1) - batch_preds.unsqueeze(0)

        # mask 用于只选择上三角矩阵（不含对角线），避免重复比较
        triu_mask = torch.triu(torch.ones_like(diff_targets, dtype=torch.bool), diagonal=1)
        # 条件：真实分数差异在一定范围内，且是上三角矩阵中的元素
        cond = (diff_targets.abs() < epsilon) & (diff_targets.abs() > min_diff) & triu_mask
        total_pairs = cond.sum().item()

        if total_pairs > 0:
            # 检查预测差异的符号是否与真实差异的符号匹配
            sign_match = (torch.sign(diff_preds) == torch.sign(diff_targets))
            correct_pairs = (sign_match & cond).sum().item()
        else:
            correct_pairs = 0

        correct += correct_pairs
        total += total_pairs

    return correct / total if total > 0 else 1.0

def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, batch_size: int = 2000) -> tuple:
    """
    计算一组标准的评估指标。
    """
    # 将张量移动到CPU并转换为NumPy数组，因为sklearn和scipy库在CPU上运行
    preds_np = preds.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    # 计算 MSE 和 MAE
    mse = mean_squared_error(targets_np, preds_np)
    mae = mean_absolute_error(targets_np, preds_np)

    # 计算 Pearson 相关系数，处理可能因输入标准差为0而导致的异常
    try:
        pearson, _ = pearsonr(targets_np, preds_np)
        if np.isnan(pearson): pearson = 0.0
    except ValueError:
        pearson = 0.0

    # 计算排序准确率
    try:
        rank_acc = local_pairwise_accuracy_torch(preds, targets, batch_size=batch_size)
    except Exception:
        rank_acc = 0.0

    return mse, mae, pearson, rank_acc

def make_pairwise_loss(loss_type: str = 'logistic', margin: float = 1.0, epsilon: float = 3.0, min_diff: float = 0.1):
    """
    一个工厂函数，用于创建成对排序损失（Pairwise Ranking Loss）。
    这个损失函数的目标是让模型学会正确地对样本进行排序。
    """

    def loss_fn(preded: torch.Tensor, gold: torch.Tensor, return_pair_count: bool = False):
        preded = preded.squeeze(-1)
        gold = gold.squeeze(-1)
        N = preded.size(0)
        if N < 2:
            return (preded.new_zeros([]), 0) if return_pair_count else preded.new_zeros([])
        # 与 local_pairwise_accuracy_torch 类似，创建差异矩阵
        pred_i, pred_j = preded.unsqueeze(0), preded.unsqueeze(1)
        gold_i, gold_j = gold.unsqueeze(0), gold.unsqueeze(1)
        gold_diff = gold_i - gold_j
        pred_diff = pred_i - pred_j
        # 定义需要计算损失的有效对
        close_mask = (gold_diff.abs() < epsilon) & (gold_diff.abs() > min_diff)
        upper_mask = torch.triu(torch.ones_like(close_mask, dtype=torch.bool), 1)
        mask = close_mask & upper_mask
        pair_count = mask.sum().item()
        if pair_count == 0:
            return (preded.new_zeros([]), 0) if return_pair_count else preded.new_zeros([])
        # 核心损失计算
        sign = torch.sign(gold_diff)
        if loss_type == 'logistic':
            # Logistic Loss: 目标是让 sign * pred_diff 的值尽可能大
            pair_loss = torch.nn.functional.softplus(-sign * pred_diff)
        else:  # Hinge Loss
            pair_loss = torch.clamp(margin - sign * pred_diff, min=0.0)
        loss = pair_loss[mask].mean()
        return (loss, pair_count) if return_pair_count else loss
    return loss_fn

def run_epoch(epoch, model, loader, criterion, pairwise_loss_fn, optimizer, accelerator, train=True,
              inner_batch_size=32):
    """
    执行一个完整的训练或评估周期（epoch），并支持批次内再分批以节省显存。
    """
    model.train() if train else model.eval()
    total_loss_epoch, total_rank_loss_epoch, total_rank_pair_count_epoch = 0, 0, 0
    all_preds_epoch, all_targets_epoch = [], []

    pbar = tqdm(loader, desc="Training" if train else "Evaluating", disable=not accelerator.is_main_process)


    # 1. 外层循环：遍历DataLoader提供的每个“逻辑大批次”
    for logical_batch in pbar:
        # --- 初始化一个逻辑大批次的
        # 累加器 ---
        accumulated_pointwise_loss = 0.0
        preds_for_ranking, targets_for_ranking = [], []
        logical_batch_size = len(logical_batch['query_texts'])

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            # 2. 内层循环：将逻辑大批次切分为“物理小批次”
            for i in range(0, logical_batch_size, inner_batch_size):
                start_idx, end_idx = i, min(i + inner_batch_size, logical_batch_size)
                inner_batch = {
                    'query_texts': logical_batch['query_texts'][start_idx:end_idx],
                    'triples_list_batch': logical_batch['triples_list_batch'][start_idx:end_idx],
                    'path_scores': logical_batch['path_scores'][start_idx:end_idx],
                    'last_triple_scores': logical_batch['last_triple_scores'][start_idx:end_idx],
                }

                pred_last_triple_scores = model(
                    inner_batch['triples_list_batch'], inner_batch['query_texts']
                )
                pred_last_triple_scores = pred_last_triple_scores.squeeze()
                path_scores_targets = inner_batch['path_scores']
                last_triple_scores_targets = inner_batch['last_triple_scores']

                loss2 = criterion(pred_last_triple_scores, last_triple_scores_targets)

                accumulated_pointwise_loss += 1e-5 * loss2

                preds_for_ranking.append(pred_last_triple_scores)
                targets_for_ranking.append(last_triple_scores_targets)

            # --- 所有小批次处理完毕，开始计算总损失 ---
            final_preds = torch.cat(preds_for_ranking)
            final_targets = torch.cat(targets_for_ranking)

            rank_loss, rank_pair_count = pairwise_loss_fn(
                final_preds, final_targets, return_pair_count=True
            )

            total_loss_logical_batch = accumulated_pointwise_loss + 1.0 * rank_loss

        if train:
            accelerator.backward(total_loss_logical_batch)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        # --- 收集整个epoch的统计数据 ---
        all_preds_epoch.append(final_preds.detach().cpu())
        all_targets_epoch.append(final_targets.detach().cpu())

        total_loss_epoch += total_loss_logical_batch.item() * logical_batch_size
        if rank_pair_count > 0:
            total_rank_loss_epoch += rank_loss.item() * rank_pair_count
            total_rank_pair_count_epoch += rank_pair_count


        if accelerator.is_main_process:
            # 1. 计算当前这个逻辑大批次的详细指标
            batch_mse, batch_mae, batch_pearson, batch_rank_acc = compute_metrics(
                final_preds.detach(), final_targets.detach()
            )
            # 2. 将所有指标都更新到进度条上
            pbar.set_postfix({
                'batch_loss': f"{total_loss_logical_batch.item():.4f}",
                'batch_mse': f"{batch_mse:.4f}",
                'batch_mae': f"{batch_mae:.4f}",
                'rank_loss': f"{rank_loss.item():.4f}" if rank_pair_count > 0 else "0.0000",
                'rank_pairs': rank_pair_count,
                'rank_acc': f"{batch_rank_acc:.3f}"
            })

    # --- Epoch 结束后，汇总所有进程的数据并计算最终指标 ---

    # 1. 首先，将CPU张量列表拼接起来，然后把结果统一移动到当前进程对应的GPU设备上
    preds_to_gather = torch.cat(all_preds_epoch).to(accelerator.device)
    targets_to_gather = torch.cat(all_targets_epoch).to(accelerator.device)

    # 2. 然后，使用位于GPU上的张量来执行数据汇总操作
    all_preds_gathered = accelerator.gather_for_metrics(preds_to_gather)
    all_targets_gathered = accelerator.gather_for_metrics(targets_to_gather)

    mse, mae, pearson, rank_acc = compute_metrics(all_preds_gathered, all_targets_gathered)

    total_loss_gathered = accelerator.gather(torch.tensor(total_loss_epoch).to(accelerator.device)).sum().item()
    avg_loss = total_loss_gathered / len(loader.dataset)

    total_rank_loss_gathered = accelerator.gather(
        torch.tensor(total_rank_loss_epoch).to(accelerator.device)).sum().item()
    total_rank_pair_count_gathered = accelerator.gather(
        torch.tensor(total_rank_pair_count_epoch).to(accelerator.device)).sum().item()
    avg_rank_loss = (
                total_rank_loss_gathered / total_rank_pair_count_gathered) if total_rank_pair_count_gathered > 0 else 0.0

    return avg_loss, mse, mae, pearson, avg_rank_loss, total_rank_pair_count_gathered, rank_acc


def train_model(train_data, val_data, pretrained_model='bert-base-uncased', num_epochs=15, batch_size=256, early_stop=3):
    """
    训练流程的主函数。
    """

    print(f"开始训练排序模型，实验名称: {args.exp_name}")
    # ✨ 初始化 Accelerator。这是第一步，也是最重要的一步。
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[ddp_kwargs])

    # 创建数据集和数据加载器
    train_loader = DataLoader(PathDataset(train_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(PathDataset(val_data), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 初始化模型、损失函数和优化器
    # 注意：这里不再需要 .to(device)
    model = PathScorer(pretrained_model)
    model.load_encoder('./checkpoints/encoder.pth', unfree_layers=args.unfree_layers)
    # model.load_state_dict(torch.load('checkpoints/path_scorer_model.pth'))
    criterion = torch.nn.MSELoss()
    # 只将需要训练的参数（requires_grad=True）传递给优化器
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5, weight_decay=1e-4)
    pairwise_loss_fn = make_pairwise_loss('logistic')
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader)
    best_val_loss, patience = float('inf'), 0
    # 开始训练循环
    for epoch in range(1, num_epochs + 1):
        # ✨ 只在主进程上打印 epoch 标题
        if accelerator.is_main_process:
            print(f"\n{'=' * 20} Epoch {epoch}/{num_epochs} {'=' * 20}")
        # 执行训练周期，注意传入 accelerator 对象
        train_stats = run_epoch(epoch, model, train_loader, criterion, pairwise_loss_fn, optimizer, accelerator, train=True)
        # 执行评估周期
        val_stats = run_epoch(epoch, model, val_loader, criterion, pairwise_loss_fn, None, accelerator, train=False)
        # ✨ 等待所有进程都完成当前 epoch 的计算，这是一个重要的同步点
        accelerator.wait_for_everyone()
        # 解包评估结果
        train_loss, train_mse, train_mae, train_pearson, train_rank_loss, train_rank_pair_count, train_rank_acc = train_stats
        val_loss, val_mse, val_mae, val_pearson, val_rank_loss, val_rank_pair_count, val_rank_acc = val_stats
        # ✨ 只在主进程上执行模型检查、保存和日志打印
        if accelerator.is_main_process:
            # 检查验证集损失是否改善
            if val_loss < best_val_loss:
                best_val_loss, patience = val_loss, 0
                # ✨ 使用 accelerator.unwrap_model() 来获取原始模型（去除DDP包装）
                # 这确保了保存的模型是标准的 PyTorch 模型，可以在任何地方加载
                unwrapped_model = accelerator.unwrap_model(model)
                # 使用 accelerator.save() 来安全地保存状态字典
                os.makedirs(f'./checkpoints/{args.exp_name}', exist_ok=True)
                accelerator.save(unwrapped_model.state_dict(), f'./checkpoints/{args.exp_name}/path_scorer_model.pth')
                accelerator.save(unwrapped_model.triple_encoder.state_dict(), './checkpoints/triple_encoder.pth')
                print(f"✅ Model saved at epoch {epoch} with Val Loss {val_loss:.4f}")
            else:
                patience += 1
                if patience >= early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break
            # 打印格式化的周期总结
            print(f"\n--- Epoch {epoch} Summary ---")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(
                f"  Train Metrics: MSE={train_mse:.4f}, MAE={train_mae:.4f}, Pearson={train_pearson:.4f}, RankAcc={train_rank_acc:.4f}")
            print(
                f"  Val Metrics:   MSE={val_mse:.4f}, MAE={val_mae:.4f}, Pearson={val_pearson:.4f}, RankAcc={val_rank_acc:.4f}")
            print(f"  Train RankLoss: {train_rank_loss:.4f} (on {train_rank_pair_count} pairs)")
            print(f"  Val RankLoss:   {val_rank_loss:.4f} (on {val_rank_pair_count} pairs)\n")


if __name__ == '__main__':
    # 从 JSON 文件加载数据
    samples = []
    with open('path_samples.jsonl', encoding='utf-8') as f:
        for line in tqdm(f):
            samples.append(json.loads(line))
    random.shuffle(samples)
    # 按 90/10 的比例分割训练集和验证集
    split_idx = int(len(samples) * 0.9)
    train_samples, val_samples = samples[:split_idx], samples[split_idx:]

    # 启动训练过程
    train_model(train_samples, val_samples)