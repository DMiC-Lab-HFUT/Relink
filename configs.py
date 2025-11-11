from pathlib import Path

from loguru import logger
import sys
import yaml
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='运行RVEA...')
parser.add_argument('--expand_COG', default=True, type=str2bool)
parser.add_argument('--dataset_name', default="wikiMQA", type=str, help='dataset name')
parser.add_argument('--query_setting', default="global", type=str, help=' global | local')
parser.add_argument('--rank_strategy', default="trained_ranker", type=str, help='Options: [embedding|llm|hybrid|trained_ranker|ranker_llm]')
parser.add_argument('--max_depth', default=5, type=int, help='max depth of search')
parser.add_argument('--max_width', default=5, type=int, help='max depth of search')

args = parser.parse_args()

DATASET_CONFIG_FILE = f'run_config/{args.dataset_name}.yaml'
CHECKPOINTS = 'custom_models/checkpoints/'

with open(DATASET_CONFIG_FILE, 'r', encoding='utf-8') as file:
    dataset_config = yaml.safe_load(file)
NEO4J_CONFIG = {
    "uri": dataset_config['neo4j']['graph']['uri'],
    "auth": (dataset_config['neo4j']['graph']['username'], dataset_config['neo4j']['graph']['password']),
}
FEEDBACK_BASE = NEO4J_CONFIG
OCCURRENCE_GRAPH = {
    "uri": dataset_config['neo4j']['occurrence_graph']['uri'],
    "auth": (dataset_config['neo4j']['occurrence_graph']['username'], dataset_config['neo4j']['graph']['password']),
}

RUNNING_CONFIG = {
    'max_workers': 40,
    'use_multithreading': True
}

PIPLINE_CONFIG = {
    'TCR': True,
    "QF": False,
    "expand_COG": args.expand_COG,
    'two_stage': True
}


# Parameters related to the retrieval method
RETRIEVER_CONFIG = {
    "max_width": args.max_width,
    'max_depth': args.max_depth,
    "min_similarity": 0,
    "rank_batch_size": 20, # Adjust different models appropriately; the stronger the model, the larger the batch_size.
    "rank_strategy": args.rank_strategy,  # [embedding|llm|hybrid|trained_ranker|ranker_llm]
    "sufficiency_check": True
}

# 运行时参数配置
DATASET_CONFIG = {
    "query_setting": args.query_setting,
    "dataset_name": f"{dataset_config['dataset']}",
    "domain": f"{dataset_config['dataset']}",
    "dataset_path": f"data/{dataset_config['dataset']}/",
    "output_dir": f"data/{dataset_config['dataset']}/output/",
    "dataset_file": f"data/{dataset_config['dataset']}/dataset/samples.json",
    "document_file": f"data/{dataset_config['dataset']}/dataset/documents.json",
    'document_limit': -1,
    "kg_triples_file": f"data/{dataset_config['dataset']}/output/extracted_triples.json",
    "entities_vector_store_path": f"data/{dataset_config['dataset']}/vector_stores/entities_vector_store",
    "doc_vector_store_path": f"data/{dataset_config['dataset']}/vector_stores/doc_vector_store",
    "results_store_path": f"results/{dataset_config['dataset']}/"
}


# -----------------------------
# embedding model config
# -----------------------------
EMBEDDING_CONFIG = {
    "api_key": '',
    "base_url": '',
    "model_name": 'text-embedding-3-small'
}

# -----------------------------
# LLM Config
# -----------------------------

LLM_CONFIG = {
    "endpoints": [
        {
            "api_key": "",
            "api_base_url": "",
            "candidate_models": ['deepseek-v3-0324', "deepseek-v3-0324"],
            "max_attempts": 3,
        }
    ]
}

# -----------------------------
# Redis config
# -----------------------------
REDIS_CONFIG = {
    "host": "localhost",
    "port": 6579,
    "decode_responses": True,
}


# -----------------------------
# log config
# -----------------------------
logger.remove()
logger.add(
    sys.stdout,
    level="ERROR",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level:<8}</level> | "
           "<cyan>{name:<15}</cyan>:<cyan>{function:<15}</cyan>:<cyan>{line:<4}</cyan> - "
           "<level>{message}</level>"
)



TOKEN_COUNT = {
    "input": 0,
    "output": 0
}

STATISTIC = {
    "average_depth": 0.0,
    "n_of_path": 0,
    'path_lens': []
}

import threading
file_lock = threading.Lock()
file_path = Path(f"results/{dataset_config['dataset']}/"+"path.jsonl")
file_path.parent.mkdir(parents=True, exist_ok=True)
path_container = open(file_path, 'w')
def write_to_file(data):
    with file_lock:
        path_container.write(data + '\n')
        path_container.flush()

print(f"""
Successfully read configuration file: {f'run_config/{args.dataset_name}.yaml'}
""")
print(f"""
{dataset_config['dataset']}
{PIPLINE_CONFIG}
{NEO4J_CONFIG}
{RETRIEVER_CONFIG}
""")


TRIPLE_COUNT = {
    "latent": 0,
    "total": 0
}
