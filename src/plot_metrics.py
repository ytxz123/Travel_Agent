# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: 训练曲线可视化
# 功能说明: 读取 GRPO 训练的 logging.jsonl（ms-swift 记录的训练指标），
#           绘制 loss / learning_rate / reward / reward_std / num_turns /
#           train_speed 等训练曲线，便于观察训练过程与收敛情况。
# --------------------------------------------

import sys
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 将项目根目录加入 sys.path（保持与 src/ 下其他脚本一致的约定）
PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_logging(log_path):
    """
    加载训练日志（ms-swift 的 logging.jsonl），返回 pandas DataFrame。

    参数:
        log_path: logging.jsonl 的绝对路径
    """
    df = pd.read_json(log_path, lines=True)
    return df


def plot_metric(df, column, title, figsize=(10, 4), ylabel=None):
    """
    绘制指定指标的曲线。

    参数:
        df: 训练日志 DataFrame
        column: 要绘制的列名
        title: 图标题（也是输出文件名）
        figsize: 图尺寸
        ylabel: Y 轴标签（默认与 column 相同）
    """
    plt.figure(figsize=figsize)
    plt.plot(df[column])
    plt.title(title, fontsize=14)
    plt.xlabel("step")
    plt.ylabel(ylabel if ylabel else column)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def plot_all(log_path, out_dir=None, columns=None):
    """
    一键绘制全部常用训练指标并保存。

    参数:
        log_path: logging.jsonl 路径
        out_dir: 图片保存目录（None 则不保存，直接 plt.show 展示）
        columns: 要绘制的列名列表（默认绘制 loss/learning_rate/reward 等常用指标）
    """
    df = load_logging(log_path)
    print("日志字段:", list(df.keys()))

    # 默认绘制的常用指标（若日志中存在）
    default_cols = [
        "loss",
        "learning_rate",
        "reward",
        "reward_std",
        "num_turns",
        "train_speed(s/it)",
    ]
    cols = columns if columns else default_cols

    for col in cols:
        if col not in df.columns:
            print(f"[跳过] 日志中没有字段: {col}")
            continue
        fig = plot_metric(df, col, title=col)
        if out_dir:
            pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
            out_path = pathlib.Path(out_dir) / f"{col}.png"
            fig.savefig(out_path, dpi=150)
            print(f"已保存: {out_path}")
        else:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    # GRPO 训练日志路径（基于项目根目录）
    LOG_PATH = str(
        pathlib.Path(PROJECT_ROOT)
        / "output/grpo_parser_aligned_run/v8-20260514-140934/logging.jsonl"
    )

    # 保存图片到 output/plots/（也可改为 None 以便交互式 plt.show）
    OUT_DIR = str(pathlib.Path(PROJECT_ROOT) / "output/plots")

    plot_all(LOG_PATH, out_dir=OUT_DIR)
