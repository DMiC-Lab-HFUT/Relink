from configs import *
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, Optional
from tqdm import tqdm
from utils.neo4j_operator import sanitize_label
from core.retriever import GraphRetriever
from core.generator import MyGenerator
from my_evaluate.hotpot_evaluate_v1 import eval as hotpot_eval
from core.feedback import KnowledgeFeedback
from configs import logger
import configs

def eval_on_hotpot_dataset(results: List[Dict]) -> Dict[str, Any]:
    """
    评估模型在 Hotpot 数据集上的表现。
    :param results: [(predicted_answer, gold_answer), ...]
    :return: 评测指标字典
    """
    preds = {i: result['output'] for i, result in enumerate(results) if result}
    golds = [{'_id': i, 'answer': result['answer']} for i, result in enumerate(results) if result]
    preds_res = {'answer': preds}
    return hotpot_eval(preds_res, golds)

def process_sample(
    sample: Dict[str, Any],
    retriever: GraphRetriever,
    generator: MyGenerator,
    feedbacker: KnowledgeFeedback,
    step: int
) -> Dict[str, Any]:
    """
    处理单个样本，包含检索、生成、反馈流程。
    """
    try:
        question = sample['question']

        if "Where was the composer of song Unravel born?" in question:
            question
        domain = (
            sanitize_label(question)
            if configs.DATASET_CONFIG['query_setting'] == 'local'
            else configs.DATASET_CONFIG['domain']
        )
        logger.info(f"domain: {domain}")
        # 检索阶段
        paths = retriever.retrieve(
            question,
            domain=domain,
            max_depth=configs.RETRIEVER_CONFIG['max_depth'],
            max_width=configs.RETRIEVER_CONFIG['max_width']
        )
        # 生成答案
        answer = generator.generate_answer(paths, question)
        if not answer:
            logger.warning(f"No answer generated for question: {question}")
            return None
        logger.info(f"Predict: {answer} |||| Gold: {sample['answer']}")

        # 可选：知识反馈
        if configs.PIPLINE_CONFIG.get('QF', False):
            feedbacker.feedback_to_knowledge_base(
                domain=domain, answer=answer, paths=paths, question=question, step=step
            )

        output = {
            'question': question,
            "answer": sample['answer'],
            "output": answer,
            'context': paths
        }
        return output
    except Exception as e:
        logger.error(f"Error processing sample at step {step}: {e}\n{traceback.format_exc()}")
        return None

def save_json(data: Any, path: str):
    """保存数据为 JSON 文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def run(dataset_path: str = configs.DATASET_CONFIG['dataset_file'], epoch: int = 0):
    """主运行流程"""
    with open(dataset_path, encoding='utf-8') as f:
        samples = json.load(f)

    logger.debug('Initializing retriever, generator, and feedback modules...')
    retriever = GraphRetriever()
    generator = MyGenerator()
    feedbacker = KnowledgeFeedback()
    results = []

    use_multithreading = configs.RUNNING_CONFIG.get('use_multithreading', True)
    max_workers = configs.RUNNING_CONFIG.get('max_workers', 10)

    with tqdm(total=len(samples), desc=f"Reasoning Epoch {epoch}", dynamic_ncols=True) as pbar:
        if use_multithreading:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_sample, sample, retriever, generator, feedbacker, step): step
                    for step, sample in enumerate(samples)
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)
                        pbar.update(1)
                        metrics = eval_on_hotpot_dataset(results)
                        pbar.set_postfix(metrics)
        else:
            # 顺序处理
            for step, sample in enumerate(samples):
                result = process_sample(sample, retriever, generator, feedbacker, step)
                if result:
                    results.append(result)
                    pbar.update(1)
                    metrics = eval_on_hotpot_dataset(results)
                    pbar.set_postfix(metrics)
    # 评测与结果保存
    final_metrics = eval_on_hotpot_dataset(results)
    print(final_metrics)
    results_dir = configs.DATASET_CONFIG['results_store_path']
    if configs.PIPLINE_CONFIG['expand_COG']:
        save_json(final_metrics, f"{results_dir}/metrics_epoch_{epoch}_expand_COG.json")
        save_json(results, f"{results_dir}/results_expand_COG.json")
    else:
        save_json(final_metrics, f"{results_dir}/metrics_epoch_{epoch}.json")
        save_json(results, f"{results_dir}/results.json")

if __name__ == "__main__":
    for epoch in range(1):
        run(epoch=epoch)
        print(
            f"Current maximum depth: {configs.RETRIEVER_CONFIG['max_depth']}，"
            f" Maximum breadth: {configs.RETRIEVER_CONFIG['max_width']}"
            f"LLM: {configs.LLM_CONFIG['endpoints'][0]['candidate_models']}"
        )

        print(configs.TOKEN_COUNT)
        print(configs.PIPLINE_CONFIG)
