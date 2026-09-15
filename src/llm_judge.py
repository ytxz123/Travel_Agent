# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: 模型效果综合评估
# --------------------------------------------

import asyncio
import time
import os
import json
import re
import sys
import pathlib

# 将项目根目录加入 sys.path，保证 from prompt 可导入
PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[0])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from openai import AsyncOpenAI
from dotenv import load_dotenv

from prompt import llm_judge_system_prompt 

# 加载环境变量
load_dotenv()

_API_KEY = os.environ["OPENAI_API_KEY"]
_BASE_URL = os.environ["OPENAI_BASE_URL"]
_MODEL = os.environ.get("JUDGE_MODEL_ID") or "gpt-5.4-mini"
# _MODEL = "gemini-3-flash-preview" 


class LLMJudge:
    def __init__(self, llm=None):
        if not llm:
            self.llm = AsyncOpenAI(
                base_url=_BASE_URL,
                api_key=_API_KEY,
                timeout=300
            )
        else:
            self.llm = llm

        self.system_prompt = llm_judge_system_prompt

        self.score_a_pattern = re.compile(
            r'"combined_scores"\s*:\s*\{[^{}]*?"Agent_A"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
            re.S | re.I,
        )
        self.score_b_pattern = re.compile(
            r'"combined_scores"\s*:\s*\{[^{}]*?"Agent_B"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
            re.S | re.I,
        )
        self.winner_pattern = re.compile(
            r'"winner"\s*:\s*"(?P<winner>Agent_A|Agent_B|Tie)"', re.I
        )

    async def _compute_unidirectional(
        self, prediction: list[dict], reference: list[dict], query: str
    ) -> tuple[float, float]:
        trajectory_a, answer_a = self.process_messages(prediction)
        trajectory_b, answer_b = self.process_messages(reference)
        # print("trajectory_a", trajectory_a)
        # print("answer_a", answer_a)
        # print("trajectory_b", trajectory_b)
        # print("answer_b", answer_b)

        prompt = f"""<USER_QUERY>\n{query}\n</USER_QUERY>\n\n<PATH_A>\n{trajectory_a}\n</PATH_A>\n\n<PATH_B>\n{trajectory_b}\n</PATH_B>\n\n<Answer_A>\n{answer_a}\n</Answer_A>\n\n<Answer_B>\n{answer_b}\n</Answer_B>"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = await self.llm.chat.completions.create(
            model=_MODEL,
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            seed=0,
        )
        score_a, score_b = self.get_judge_scores(response.choices[0].message.content)

        return score_a, score_b

    async def compute(
        self, prediction: list[dict], reference: list[dict], query: str
    ) -> dict[str, float]:
        # 双向打分，消除位置的bias
        results = await asyncio.gather(
            self._compute_unidirectional(prediction, reference, query=query),
            self._compute_unidirectional(reference, prediction, query=query),
        )

        # 这里要注意顺序
        score_prediction = (results[0][0] + results[1][1]) / 2
        score_reference = (results[0][1] + results[1][0] ) / 2

        return {"query:": query, "prediction": round(score_prediction, 2), "reference": round(score_reference, 2)}

    def process_messages(self, messages: list[dict]) -> tuple[list[dict], str]:
        step_idx = 0
        trajectory = []
        for message in messages[:-1]:
            if message["role"] != "assistant":
                continue

            if "tool_calls" in message:
                trajectory.append(
                    {
                        "step": step_idx,
                        "tool_calls": message.get("tool_calls", ""),
                    }
                )
            else:
                content = re.sub(r'<think>.*?</think>', '', message["content"], flags=re.DOTALL).strip()
                if content:
                    trajectory.append(
                        {
                            "step": step_idx,
                            "tool_calls": content,
                        }
                    )
            step_idx += 1

        answer = "未回复"
        if messages[-1]["role"] == "assistant":
            answer = messages[-1].get("content") or answer
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()

        return trajectory, answer

    def get_judge_scores(self, response: str) -> tuple[float, float]:
        match_a = self.score_a_pattern.search(response)
        match_b = self.score_b_pattern.search(response)

        if not (match_a and match_b):
            raise ValueError(f"Failed to get judge scores in response: {response}")

        score_a = float(match_a.group(1))
        score_b = float(match_b.group(1))

        return score_a, score_b


    def chunk_by_size(self, lst, chunk_size):
        """将列表按固定大小拆分"""
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


    async def process_batch(self, prediction, reference, query, chunk_task_size=20):
        """并发处理所有数据"""
        tasks = [self.compute(p, r, q) for p, r, q in zip(prediction, reference, query)]
        split_tasks = self.chunk_by_size(tasks, chunk_task_size)
        final_results = []
        for idx, single_task in enumerate(split_tasks):
            results = await asyncio.wait_for(
                asyncio.gather(*single_task),
                timeout=300
            )
            final_results.extend(results)
            print(f"task_{idx} finished, processed {len(results)} samples.")
        return final_results


if __name__ == "__main__":
    # 项目根目录（文件系统根，= src 的父目录）
    PROJECT_ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[1])
    INFER_DIR = os.path.join(PROJECT_ROOT_DIR, "output/infer")

    # 对照组
    base = {}
    #with open(os.path.join(INFER_DIR, "output_qwen3_14b.jsonl")) as fd:
    with open(os.path.join(INFER_DIR, "output_qwen3_4b_instruct_baseline.jsonl")) as fd:
        for line in fd:
            info = json.loads(line)
            base[info["query"]] = info

    # 实验组
    pred = {}
    #with open(os.path.join(INFER_DIR, "output_qwen3_4b_sft420.jsonl")) as fd:
    with open(os.path.join(INFER_DIR, "output_qwen3_4b_grpo_epoch150.jsonl")) as fd:
        for line in fd:
            info = json.loads(line)
            pred[info["query"]] = info
 
    tasks = []
    predictions, references, queries = [], [], []
    for key, value in pred.items():
        if key in base:
            query = base[key]["query"]
            reference = base[key]["messages"]
            prediction = pred[key]["messages"]
            predictions.append(prediction)
            queries.append(query)
            references.append(reference)

    print(len(predictions), len(references))
    llm_judge = LLMJudge()
    result = asyncio.run(llm_judge.process_batch(predictions, references, queries))
    for r in result:
        print(r)

    ref_scores = [r["reference"] for r in result]
    rl_scores = [r["prediction"] for r in result]
    print("judged samples: ", len(ref_scores))
    print("baseline mean: ", sum(ref_scores) / len(ref_scores))
    print("predict mean: ", sum(rl_scores) / len(ref_scores))

