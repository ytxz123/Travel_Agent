#!/usr/bin/env bash
set -euo pipefail

# 项目根目录：脚本位于 <根>/scripts/ 下，上提一级得到项目根。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 基座模型与 SFT 产出的 LoRA adapter（按实际训练输出目录修改）。
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/model/Qwen/Qwen3-4B-Instruct-2507}"
ADAPTERS="${ADAPTERS:-${PROJECT_ROOT}/output/qwen3_4b_sft_4gpu/v8-20260511-142452/checkpoint-420}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/qwen3_4b_sft_4gpu/v8-20260511-142452/qwen3_4b_sft_merged_420}"

swift export \
    --model "${MODEL_PATH}" \
    --adapters "${ADAPTERS}" \
    --output_dir "${OUTPUT_DIR}" \
    --merge_lora true \
    --safe_serialization true
