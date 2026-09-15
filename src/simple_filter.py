# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: 蒸馏轨迹粗清洗（simple_filter）
# 功能说明: 读取 output/train_expansion/ 与 output/test/ 下的蒸馏轨迹 JSON，
#           按规则过滤异常样本（轮数超限/未调用工具/无回答/全英文/回答过短），
#           将保留样本转换为 {id, question, answer} 结构并写出 data/clean/ 下的 jsonl。
# --------------------------------------------

import os
import re
import json
import sys
import pathlib
import numpy as np

from langdetect import detect

# 将项目根目录加入 sys.path（保持与 src/ 下其他脚本一致的约定）
PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def filter_trajectories(save_dir):
    """
    读取指定目录下的所有蒸馏轨迹 JSON，按规则过滤并统计。

    参数:
        save_dir: 蒸馏轨迹所在目录（如 output/train_expansion 或 output/test）

    返回:
        (过滤后的样本列表, 各工具调用次数统计)
    """
    files = os.listdir(save_dir)
    tool_calls_stat = {}   # 统计每个工具被调用次数
    dropped_count = 0      # 被过滤掉的样本数
    total_count = 0        # 保留的样本数
    data = []              # 保留的样本
    stat_turns = []        # 保留样本的对话轮数（用于分位数统计）

    for filename in files:
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(save_dir, filename), encoding="utf-8") as fd:
            info = json.load(fd)

        question = info["question"]
        msg = info["messages"][-1]["content"]      # 最后一轮助手回答
        stats = info["stats"]                       # 工具调用统计

        # 从 stats 中取出对话轮数，剩余字段即各工具调用次数
        turn = stats.pop("turns", 0)
        num_calls = sum(stats.values())

        # ---- 过滤规则 ----
        # 1. 轮数超过 13 或完全没调用工具
        if turn > 13 or num_calls == 0:
            dropped_count += 1
            continue
        # 2. 触发"超出对话轮数/上下文最大长度"的系统提示
        elif "exceeds the limit" in msg or "reached the maximum" in msg:
            dropped_count += 1
            continue
        # 3. 回答里含"没有找到"，说明该 query 本身信息不足
        elif "没有找到" in msg:
            dropped_count += 1
            continue
        # 4. 回答被识别为英文（期望中文回答）
        elif detect(msg) == "en":
            dropped_count += 1
            continue
        # 5. 回答太短
        elif len(msg) < 100:
            dropped_count += 1
            continue

        # 记录各工具调用次数
        for name, value in stats.items():
            tool_calls_stat[name] = tool_calls_stat.get(name, 0) + 1

        # 构造清洗后的样本（answer 保存原始完整 messages 的 JSON 字符串）
        new_info = {
            "qid": filename.split(".")[0],
            "question": info["question"],
            "answer": json.dumps(info["messages"], ensure_ascii=False),
        }
        data.append(new_info)
        stat_turns.append(turn)
        total_count += 1

    # 打印工具调用统计
    for k, v in tool_calls_stat.items():
        print(k, v)
    print()
    print(f"total count: {total_count}, dropped count: {dropped_count}")
    return data, stat_turns


def print_turn_percentiles(stat_turns):
    """打印保留样本对话轮数的分位数统计。"""
    for k in [50, 60, 70, 80, 90, 95]:
        print(f"{k}分位: {np.percentile(stat_turns, k)}")


def convert_to_conversations(data, output_path):
    """
    将 {qid, question, answer} 样本转换为 {id, conversations} 格式并写出 jsonl。

    转换规则（与 notebook 保持一致）:
        - system 消息：将"最多调用50轮工具"统一改写为"最多调用13轮工具"
        - assistant 消息：合并 reasoning_details 为 <think> 块；
          有 tool_calls 则包装为 <tool_call>，否则保留原文
        - tool 消息：包装为 <tool_response> 作为 user 角色
    """
    # 确保输出目录存在
    save_dir = os.path.dirname(output_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_data = []
    with open(output_path, "w", encoding="utf-8") as handler:
        for item in data:
            messages = json.loads(item["answer"])
            conversations = []
            for message in messages:
                role = message["role"]

                if role == "system":
                    # 统一工具轮数限制：50 -> 13
                    content = message["content"].replace("最多调用50轮工具", "最多调用13轮工具")
                    info = {"role": "system", "content": content}

                elif role == "user":
                    info = {"role": "user", "content": message["content"]}

                elif role == "assistant":
                    # 合并推理内容到 <think> 块
                    reasoning_body = message["reasoning_details"]
                    reasoning_content = "\n".join(
                        [reasoning_body[t]["text"] for t in range(len(reasoning_body))]
                    )
                    if reasoning_content.strip():
                        reasoning_content = f"<think>\n{reasoning_content}\n</think>\n\n"
                    else:
                        reasoning_content = ""

                    if message["tool_calls"]:
                        # 工具调用轮：包装为 <tool_call>
                        content = json.dumps(message["tool_calls"], ensure_ascii=False)
                        info = {
                            "role": "assistant",
                            "content": f"{reasoning_content}<tool_call>\n{content}\n</tool_call>",
                        }
                    elif message["content"].strip():
                        # 最终回答轮：保留原文（可能已含 <answer>）
                        content = message["content"]
                        info = {
                            "role": "assistant",
                            "content": f"{reasoning_content}{content}",
                        }

                elif role == "tool":
                    # 工具返回结果：包装为 <tool_response> 并作为 user 角色
                    info = {
                        "role": "user",
                        "content": f"<tool_response>\n{message['content']}\n</tool_response>",
                    }

                conversations.append(info)

            body = {"id": item["qid"], "conversations": conversations}
            all_data.append(body)
            handler.write(json.dumps(body, ensure_ascii=False) + "\n")

    print(f"转换完成：共 {len(all_data)} 条，输出到 {output_path}")
    return all_data


if __name__ == "__main__":
    # ---- 1. 清洗训练集蒸馏轨迹 ----
    train_data, train_turns = filter_trajectories(
        os.path.join(PROJECT_ROOT, "output/train_expansion")
    )
    print_turn_percentiles(train_turns)

    # ---- 2. 清洗测试集蒸馏轨迹 ----
    test_data, test_turns = filter_trajectories(
        os.path.join(PROJECT_ROOT, "output/test")
    )
    print_turn_percentiles(test_turns)

    # ---- 3. 转换为 {id, conversations} 格式并写出 ----
    train_out = os.path.join(PROJECT_ROOT, "data/clean/train.jsonl")
    test_out = os.path.join(PROJECT_ROOT, "data/clean/test.jsonl")

    all_data = convert_to_conversations(train_data, train_out)
    convert_to_conversations(test_data, test_out)

    # 随机抽一条查看（可选）
    # print(random.choice(all_data)["conversations"])
