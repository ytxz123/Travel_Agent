# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: 数据/输出样例查看
# 功能说明: 快速查看一条训练数据样本（conversations 结构）
#           或一条模型推理输出样本（query / prediction），便于人工核验。
# --------------------------------------------

import json
import sys
import pathlib

# 将项目根目录加入 sys.path（保持与 src/ 下其他脚本一致的约定）
PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def view_dataset_sample(dataset_path, max_items=None):
    """
    查看训练数据样本的 conversations 结构。

    参数:
        dataset_path: 训练数据 jsonl 路径
        max_items: 最多查看的样本数；None 表示只看 1 条
    """
    print(f"===== 数据集样本: {dataset_path} =====")
    count = 0
    with open(dataset_path, encoding="utf-8") as fd:
        for line in fd:
            info = json.loads(line)
            for item in info["conversations"]:
                print("role: ", item["role"])
                print("content: ", item["content"])
                print("=" * 100)
            count += 1
            if max_items is not None and count >= max_items:
                break
    print(f"(共显示 {count} 条样本)")


def view_infer_sample(infer_path, max_lines=None):
    """
    查看模型推理输出样本。

    参数:
        infer_path: 推理结果 jsonl 路径（含 query / prediction 字段）
        max_lines: 最多查看的行数；None 表示全部
    """
    print(f"===== 推理输出样本: {infer_path} =====")
    lines_read = 0
    with open(infer_path, encoding="utf-8") as fd:
        for line in fd:
            info = json.loads(line)
            print("query: ", info["query"])
            print("answer: ", info["prediction"])
            print("=" * 100)
            lines_read += 1
            if max_lines is not None and lines_read >= max_lines:
                break
    print(f"(共显示 {lines_read} 行)")


if __name__ == "__main__":
    # 默认查看一条已清洗的训练数据样本（test 集）
    view_dataset_sample(
        str(pathlib.Path(PROJECT_ROOT) / "data/clean/test_clean_openai_toolcalls.jsonl"),
        max_items=1,
    )

    # 也可查看模型推理输出（取消注释使用）：
    # view_infer_sample(str(pathlib.Path(PROJECT_ROOT) / "output/infer/output_qwen3_4b_instruct_baseline.jsonl"))
    # view_infer_sample(str(pathlib.Path(PROJECT_ROOT) / "output/infer/output_qwen3_4b_sft420.jsonl"))
