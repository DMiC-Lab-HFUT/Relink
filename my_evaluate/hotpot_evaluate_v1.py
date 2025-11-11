import sys
import ujson as json
import re
import string
from collections import Counter
import pickle


def normalize_answer(s):
    """
    规范化答案，包括去除文章、空格修复、去除标点和转换为小写。
    """

    def remove_articles(text):
        # 去除文章（a, an, the）
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        # 修复多余的空格
        return ' '.join(text.split())

    def remove_punc(text):
        # 去除标点符号
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        # 转换为小写
        return text.lower()

    # 依次应用各个处理步骤
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """
    计算F1分数、精确度和召回率。
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)  # 指标为零的情况

    # 特殊情况处理：如果预测和真实答案是“yes”、“no”或“noanswer”，且不相等，返回零指标
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    # 将预测和真实答案分词
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)  # 计算共同词
    num_same = sum(common.values())  # 共同词的数量

    if num_same == 0:
        return ZERO_METRIC  # 如果没有共同词，返回零指标

    # 计算精确度和召回率
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)  # 计算F1分数

    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    """
    计算准确匹配（EM）分数。
    """
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def update_answer(metrics, prediction, gold):
    """
    更新答案的指标，包括EM、F1、精确度和召回率。
    """
    em = exact_match_score(prediction, gold)  # 计算EM
    f1, prec, recall = f1_score(prediction, gold)  # 计算F1、精确度和召回率

    # 更新指标
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall


def eval(prediction, gold):
    """
    评估预测和真实答案，计算各项指标。
    """

    # 初始化指标字典
    metrics = {
        'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
    }

    # 遍历每个数据点
    for dp in gold:
        cur_id = dp['_id']  # 当前数据点的ID

        # 更新答案指标
        if cur_id not in prediction['answer']:
            print('missing answer {}'.format(cur_id))
        else:
            em, prec, recall = update_answer(
                metrics, prediction['answer'][cur_id], dp['answer'])

    # 计算每个指标的平均值
    N = len(gold)
    for k in metrics.keys():
        metrics[k] /= N

    # 输出最终指标
    return metrics


def eval_on_hotpot_dataset(res):
    preds = {}
    golds = []
    for i, answer in enumerate(res):

        if not answer:
            continue
        pred, gold = answer
        preds[i] = answer[0]
        golds.append({'_id': i, 'answer': answer[1]})

    preds_res = {}
    preds_res['answer'] = preds
    # 从命令行参数获取预测和真实答案文件路径
    return eval(preds_res, golds)


if __name__ == '__main__':

    with open('../results/2wikiMultihopQA/beko/BEKO_res_round1_.json', encoding='utf-8') as f:
        res = json.load(f)
    preds = {}
    golds = []
    for i, answer in enumerate(res):

        if not answer:
            continue
        pred, gold = answer
        preds[i] = answer[0]
        golds.append({'_id': i, 'answer': answer[1]})

    preds_res = {}
    preds_res['answer'] = preds

    # 从命令行参数获取预测和真实答案文件路径
    metrics = eval(preds_res, golds)
    print(metrics)
