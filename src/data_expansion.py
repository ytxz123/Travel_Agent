# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: 种子问题扩写（LLM 改写扩充）
# 功能说明: 读取 data/train_seed.jsonl 中的种子旅行问题，
#           用 LLM 将每个问题改写出 5 个新问题（替换地点/时间/约束等），
#           并发处理、去重后写入 data/train_expansion.jsonl。
# --------------------------------------------

import os
import re
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import version

from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm

# 将项目根目录加入 sys.path（保持与 src/ 下其他脚本一致的约定）
import sys
import pathlib
PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 加载环境变量（OPENAI_API_KEY / OPENAI_BASE_URL）
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# 固定随机种子，保证结果可复现
random.seed(42)


# ---------------------------------------------------------------------------
# 1. 环境与提示词
# ---------------------------------------------------------------------------
def print_versions():
    """打印关键依赖版本，便于排查环境问题。"""
    print(f"openai version:  {version('openai')}")
    print(f"python-dotenv version:  {version('python-dotenv')}")


# 问题扩写提示词：让模型基于用户问题改写生成 5 个新问题
QUERY_EXPANSION_PROMPT = """
你是一个旅行经验丰富的攻略改写专家，对中国各大城市、交通、路线等非常熟悉。请根据下面给出的用户旅行规划的问题，生成5个新的问题。

请遵守下面的一些改写要求：
1.新用户问题只是参考，需要是全新的问题，不能跟原来的问法的思路一样，也不能一样的内容。
2.新问题的涉及的地点，城市等需要是真实的，并且是符合逻辑的，可以通过一些常识知识中进行改写，替换掉原有的地点/城市。
3.(可选)如果必要的话，可以在新问题中加上一些约束条件，例如价格，距离，天数，交通方式，预算，途经点等。

请直接输出改写后的句子，每一个问题占一行，不要有编号或者其他无关内容。

用户问题：{}
"""


# ---------------------------------------------------------------------------
# 2. LLM 客户端
# ---------------------------------------------------------------------------
def build_client():
    """根据 .env 配置创建 OpenAI 兼容客户端。"""
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )


# ---------------------------------------------------------------------------
# 3. 单次扩写（带重试）
# ---------------------------------------------------------------------------
def chat(client, prompt, max_retry=3):
    """
    对单个用户问题调用 LLM 进行扩写，最多重试 max_retry 次。

    返回:
        (原始问题, LLM 返回的扩写文本)；失败且重试耗尽时返回 None。
    """
    def do_chat(text):
        # 将用户问题填入扩写提示词
        filled_prompt = QUERY_EXPANSION_PROMPT.format(text)
        completions = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": "你是一个乐于助人的智能助手。"},
                {"role": "user", "content": filled_prompt},
            ],
        )
        return text, completions.choices[0].message.content

    while max_retry > 0:
        try:
            return do_chat(prompt)
        except Exception as e:
            max_retry -= 1
            # 失败后随机休眠 1-4 秒再重试，避免触发限流
            time.sleep(random.randint(1, 4))
    return None


# ---------------------------------------------------------------------------
# 4. 批量扩写主流程
# ---------------------------------------------------------------------------
def expand_queries(client, input_path, output_path, max_workers=50):
    """
    读取种子问题，并发调用 LLM 扩写，去重后写出 jsonl。

    参数:
        client: OpenAI 兼容客户端
        input_path: 种子问题 jsonl 路径（每行 {"question": "..."}）
        output_path: 扩写结果 jsonl 输出路径
        max_workers: 并发线程数（默认 50，可根据大模型接口 QPS 调整）
    """
    # 读取种子问题
    queries = []
    with open(input_path, encoding="utf-8") as fd:
        for idx, line in enumerate(fd):
            info = json.loads(line)
            queries.append((idx, info["question"]))

    # 并发调用 LLM 扩写每个问题
    expand_queries = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {idx: executor.submit(chat, client, query) for idx, query in queries}
        for idx in tqdm(futures):
            future = futures[idx]
            result = future.result()
            if result is None:
                continue  # 重试失败则跳过该问题
            query, expand = result
            # 按行拆分扩写结果，去除"1. "、"2. " 之类的编号前缀
            expand = expand.split("\n")
            expand = [re.sub(r"\d. ", "", item) for item in expand if item.strip()]
            expand_queries.append(query)
            expand_queries.extend(expand)

    # 去重（保留原始顺序）
    expand_queries = list(set(expand_queries))

    # 写出结果：每行一个 {"question": "..."}
    with open(output_path, "w", encoding="utf-8") as fw:
        for query in expand_queries:
            info = {"question": query}
            fw.write(json.dumps(info, ensure_ascii=False) + "\n")

    print(f"扩写完成：共 {len(expand_queries)} 条问题，输出到 {output_path}")


# ---------------------------------------------------------------------------
# 5. 主入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print_versions()

    # 创建 LLM 客户端
    client = build_client()

    # 数据路径：基于项目根目录精确定位
    INPUT_PATH = os.path.join(PROJECT_ROOT, "data/train_seed.jsonl")
    OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/train_expansion.jsonl")

    # 单样本测试（可注释掉跳过）
    # test_query = "一家人在国庆到澳门3日跟团游，行程舒适为主，含历史文化和小吃美食，预算约6000元，推荐几楼？"
    # print(chat(client, test_query))

    # 批量扩写
    expand_queries(client, INPUT_PATH, OUTPUT_PATH, max_workers=50)
