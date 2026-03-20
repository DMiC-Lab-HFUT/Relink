import asyncio
import json
import logging
import time
import uuid
import argparse
from contextlib import asynccontextmanager
from typing import List
from path_scorer import *
import redis
import torch
import uvicorn
from fastapi import FastAPI
from tqdm import tqdm

# 0721
# --- 1. 配置参数 ---
MODEL_PATH = 'checkpoints/scorer_best.pth'  # 您的模型路径
BATCH_SIZE = 128
REDIS_HOST = '114.213.211.73'
REDIS_PORT = 6779
REDIS_DB = 0
REQUEST_STREAM = 'request_stream'
RESPONSE_STREAM = 'response_stream'
CONSUMER_GROUP = 'scorer_workers'

# --- 2. 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- 3. 全局变量 ---
trained_score_model = None
redis_client = None
device = None


# --- 4. Lifespan 事件处理器 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global trained_score_model, redis_client, device
    gpu_id = app.state.gpu_id
    log_prefix = f"[GPU-{gpu_id}]"
    logger.info(f"{log_prefix} 应用启动中...")

    # 设置计算设备
    if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')
        if torch.cuda.is_available():
            logger.warning(f"{log_prefix} CUDA GPU {gpu_id} 不可用, 将使用 CPU。")
        else:
            logger.info(f"{log_prefix} 未找到可用 CUDA 设备, 将使用 CPU。")

    # 初始化 Redis (注意: 不使用 decode_responses=True，因为 Stream 命令处理字节流更稳定)
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    try:
        redis_client.ping()
        logger.info(f"{log_prefix} 成功连接到 Redis: {REDIS_HOST}:{REDIS_PORT}")
    except redis.exceptions.ConnectionError as e:
        logger.error(f"{log_prefix} 无法连接到 Redis: {e}")
        raise

    # 加载模型
    logger.info(f"{log_prefix} 正在加载模型...")
    trained_score_model = PathScorer()  # 使用模拟模型
    trained_score_model.load_state_dict(torch.load(MODEL_PATH, map_location=device)) # 当您使用真实模型时，请取消此行注释
    trained_score_model.to(device)
    trained_score_model.eval()
    logger.info(f"{log_prefix} 模型加载完毕，正在使用设备: {device}")

    # 关键步骤: 创建消费者组 (由一个进程负责即可)
    if gpu_id == 0:
        try:
            redis_client.xgroup_create(REQUEST_STREAM, CONSUMER_GROUP, id='0', mkstream=True)
            logger.info(f"{log_prefix} 成功创建或确认消费者组 '{CONSUMER_GROUP}' 在流 '{REQUEST_STREAM}' 上。")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"{log_prefix} 消费者组 '{CONSUMER_GROUP}' 已存在。")
            else:
                logger.error(f"{log_prefix} 创建消费者组时发生 Redis 错误: {e}")
                raise e

    # 为每个工作进程启动一个后台任务
    consumer_name = f"gpu-{gpu_id}-{uuid.uuid4().hex[:6]}"
    asyncio.create_task(redis_worker(log_prefix, consumer_name))

    yield

    # --- 应用关闭逻辑 ---
    logger.info(f"{log_prefix} 应用关闭中...")
    if redis_client:
        redis_client.close()
        logger.info(f"{log_prefix} Redis 连接已关闭。")


# --- 5. 创建 FastAPI 应用 ---
app = FastAPI()
# --- 6. 后台处理任务 ---
async def redis_worker(log_prefix: str, consumer_name: str):
    logger.info(f"{log_prefix} 后台任务 ({consumer_name}) 已启动，正在监听流 '{REQUEST_STREAM}'...")
    global last_log_time
    while True:
        try:
            pending_info = redis_client.xpending(REQUEST_STREAM, CONSUMER_GROUP)
            pending_count = pending_info['pending']
            logger.info(f"{log_prefix} 当前等待队列长度: {pending_count}")

            response = redis_client.xreadgroup(
                groupname=CONSUMER_GROUP,
                consumername=consumer_name,
                streams={REQUEST_STREAM: '>'},
                count=1,
                block=0  # 永久阻塞
            )
            if not response: continue

            stream_name, messages = response[0]
            message_id, data = messages[0]

            correlation_id = None
            try:
                correlation_id = data[b'correlation_id'].decode('utf-8')
                payload = json.loads(data[b'payload'].decode('utf-8'))
                paths = payload['paths']
                query = payload['query']

                req_log_prefix = f"{log_prefix}[ReqID: {correlation_id[:8]}]"
                logger.info(f"{req_log_prefix} 收到 {len(paths)} 条路径的新请求。")
                start_time = time.time()

                all_scores = []
                queries = [query] * len(paths)
                is_cuda = (device.type == 'cuda')

                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=is_cuda):
                    pbar = tqdm(total=len(paths), desc=f"推理中 {req_log_prefix}", leave=False, ncols=100)
                    for i in range(0, len(paths), BATCH_SIZE):
                        batch_paths = paths[i:i + BATCH_SIZE]
                        batch_queries = queries[i:i + BATCH_SIZE]
                        last_triple_scores = trained_score_model(batch_paths, batch_queries)
                        scores = last_triple_scores.squeeze(-1).cpu().tolist()
                        if isinstance(scores, float): scores = [scores]
                        all_scores.extend(scores)
                        pbar.update(len(batch_paths))
                    pbar.close()

                response_payload = json.dumps({"result": all_scores})
                response_message = {
                    'correlation_id': correlation_id,
                    'result': response_payload
                }
                redis_client.xadd(RESPONSE_STREAM, response_message)
                redis_client.xack(REQUEST_STREAM, CONSUMER_GROUP, message_id)
                duration = time.time() - start_time
                logger.info(f"{req_log_prefix} 请求处理完毕并已确认 (ACK)，耗时 {duration:.2f} 秒。")

            except Exception as e:
                logger.exception(f"处理消息 {message_id.decode()} 时发生未知错误: {e}")
                if correlation_id:
                    error_payload = json.dumps({"error": f"处理时发生内部错误: {str(e)}"})
                    redis_client.xadd(RESPONSE_STREAM, {'correlation_id': correlation_id, 'result': error_payload})
                # 即使出错也需要确认，防止坏消息反复被读取。
                # 在生产环境中，可以设计更复杂的“死信队列”来处理失败的消息。
                redis_client.xack(REQUEST_STREAM, CONSUMER_GROUP, message_id)


        except Exception as e:
            logger.exception(f"{log_prefix} ({consumer_name}) redis_worker 主循环发生严重错误: {e}")
            await asyncio.sleep(5)


# --- 7. 主程序入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Path Scorer Multi-GPU Inference Server using Redis Streams")
    parser.add_argument("--gpu-id", type=int, default=0, help="要绑定的GPU设备的ID")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务绑定的主机地址")
    parser.add_argument("--port", type=int, default=5500, help="服务绑定的端口")
    args = parser.parse_args()

    app.state.gpu_id = args.gpu_id
    app.router.lifespan_context = lifespan

    uvicorn.run(app, host=args.host, port=args.port)


"""
cd code/RVEA/custom_models/
conda activate RVEA
tmux

python path_score_server.py --gpu-id 0 --port 5500
python path_score_server.py --gpu-id 0 --port 5501
python path_score_server.py --gpu-id 1 --port 5502
python path_score_server.py --gpu-id 1 --port 5503
python path_score_server.py --gpu-id 2 --port 5504
python path_score_server.py --gpu-id 2 --port 5505
python path_score_server.py --gpu-id 3 --port 5506
python path_score_server.py --gpu-id 3 --port 5507
"""

