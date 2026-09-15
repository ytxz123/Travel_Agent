#!/usr/bin/env bash
set -euo pipefail

# 项目根目录：脚本位于 <根>/scripts/ 下，上提一级得到项目根。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 待评估模型目录，可换成任意 checkpoint / merged 模型 / 基座模型：
#   MODEL_DIR="${PROJECT_ROOT}/model/Qwen/Qwen3-4B-Instruct-2507"
#   MODEL_DIR="${PROJECT_ROOT}/output/qwen3_4b_sft_4gpu/<run>/qwen3_4b_sft_merged_420"
#   MODEL_DIR="${PROJECT_ROOT}/output/grpo_parser_aligned_run/<run>/checkpoint-150"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/output/grpo_parser_aligned_run/v8-20260514-140934/checkpoint-150}"

OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/output/infer/output_qwen3_4b_grpo_epoch150.jsonl}"

python "${PROJECT_ROOT}/src/run_tool_loop_infer.py" \
  --infer_backend transformers \
  --dataset_path "${PROJECT_ROOT}/data/final/test_final.jsonl" \
  --model_dir "${MODEL_DIR}" \
  --tools_dir "${PROJECT_ROOT}/src/tools/" \
  --start_idx 0 \
  --num_samples 80 \
  --max_turns 13 \
  --max_new_tokens 5000 \
  --temperature 0.2 \
  --top_p 0.95 \
  --top_k 50 \
  --tool_first_enforce \
  --system_max_tool_calls 13 \
  --max_same_tool_call_rounds 3 \
  --save_full_messages \
  --output_path "${OUTPUT_PATH}"
