# Travel Agentic RL

基于 [ms-swift](https://github.com/modelscope/ms-swift) 的**旅行规划 Agent 强化学习**项目。通过 **LoRA SFT → GRPO** 两阶段训练，让模型学会自主规划工具调用、多轮推理，并输出可落地的旅行攻略。

- **基座模型**：Qwen3-4B-Instruct-2507（可换 Qwen3-14B）
- **工具链**：8 个可调用工具，接入高德地图 / FireCrawl / Tavily 真实数据
- **RL 插件**：自定义多轮工具调度器 + 课程式综合奖励函数
- **输出协议**：`<think>` / `<tool_call>` / `<tool_response>` / `<answer>`

---

## 项目概况

给定一句自然语言旅行需求（例如 *"预算有限，如何规划一条包含洪崖洞和长江索道的经济路线？"*），模型需要：

1. 自行决定先查什么、用哪个工具；
2. 多轮调用工具获取事实（景点、天气、车票、路线、周边 POI）；
3. 基于真实工具结果生成时间、交通、费用、注意事项完整的方案；
4. 严格遵守统一的标签输出协议。

训练分两阶段：

| 阶段 | 方法 | 目的 | 脚本 |
|------|------|------|------|
| SFT | LoRA 微调 | 学会工具调用格式与基本 Agent 行为 | `scripts/train_sft.sh` |
| RL | GRPO 全参微调 | 用过程奖励 + 结果奖励 + LLM Judge 提升调用与答案质量 | `scripts/train_rl.sh` |

完整链路：

```
种子问题 ──> 问题扩充 ──> 强模型蒸馏轨迹 ──> 粗清洗 ──> 格式清洗 ──> LLM 质检 ──> 最终数据集
  │                                                        │
  └── src/data_expansion.py                                └── src/quality_filter.py
                                                                    │
                          ┌─────────────────────────────────────────┘
                          ▼
              SFT (LoRA) ──> 合并权重 ──> GRPO (vLLM rollout) ──> 推理评估
           scripts/train_sft.sh   scripts/merge_lora.sh    scripts/train_rl.sh
                                                            scripts/rollout.sh
```

---

## 目录结构

```
Travel_Agentic_RL/
├── src/                      # 全部 Python 代码
│   ├── prompt.py             #   所有提示词（Agent / 抽取 / 模拟 / 质检 / 评估）
│   ├── data_expansion.py     #   数据管线①：种子问题 LLM 改写扩充
│   ├── data_distill.py       #   数据管线②：强模型 + 真实工具，蒸馏多轮轨迹
│   ├── simple_filter.py      #   数据管线③：粗清洗（语言 / 轮数 / 长度）
│   ├── format_filter.py      #   数据管线④：标签 → OpenAI tool_calls，规则过滤
│   ├── quality_filter.py     #   数据管线⑤：LLM-as-Judge 五维质量打分
│   ├── run_tool_loop_infer.py#   推理引擎：多轮真实工具循环
│   ├── llm_judge.py          #   模型对比评估（双向 LLM-Judge）
│   ├── plot_metrics.py       #   训练曲线可视化
│   ├── sample_data.py        #   数据 / 输出样例查看
│   ├── tools/                #   8 个 Agent 工具（qwen-agent BaseTool）
│   └── utils/                #   日志 / 文本截断 / JSON→Markdown
│
├── plugins/                  # ⭐ RL 核心插件（ms-swift external_plugins）
│   ├── tooluse_multi_turn_scheduler.py  # 多轮工具调度器 travel_tool_loop
│   └── tooluse_reward_parser_aligned.py # 综合奖励 external_parser_aligned_curriculum_reward
│
├── scripts/                  # 训练 / 推理 shell 脚本
│   ├── train_sft.sh          #   SFT 训练
│   ├── merge_lora.sh         #   LoRA 合并
│   ├── rollout.sh            #   启动 vLLM rollout server
│   ├── train_rl.sh           #   GRPO 强化学习训练
│   └── infer.sh              #   推理评估
│
├── .env.example              # 环境变量模板（复制为 .env）
└── requirements.txt
```

---

## 快速开始

### 1. 环境

- Python 3.10+，CUDA GPU（SFT 建议 4 卡；GRPO 建议 6 卡训练 + 2 卡 rollout）
- 训练脚本为 Bash，请在 **Linux / macOS / WSL / Git Bash** 下运行

```bash
conda create -n travel python=3.10 -y
conda activate travel

# 安装训练框架（本项目的 plugins/ 依赖 swift 包）
pip install ms-swift

# 按 CUDA 版本安装 PyTorch / vLLM（GRPO 的 rollout 依赖 vLLM）
pip install -r requirements.txt
```

### 2. 配置密钥

```bash
cp .env.example .env
# 填写 OPENAI_API_KEY / AMAP_MAPS_API_KEY / FIRECRAWL_API_KEY 等
```

### 3. 下载基座模型

放入 `model/Qwen/Qwen3-4B-Instruct-2507/`（ModelScope 或 HuggingFace 均可），然后按需修改脚本中的 `MODEL_PATH`。

### 4. 准备数据

训练数据为 jsonl，每行一条样本，字段为 `id` + `conversations`。`conversations` 是一条完整的多轮工具交互轨迹：

```json
{
  "id": "7422f7b075a69ed7f8648f06564279db",
  "conversations": [
    {
      "role": "system",
      "content": "你是旅行规划助手，需要先用工具获取事实，再给最终回答。\n每一轮只能二选一输出…\n<tools>…工具 JSON Schema…</tools>"
    },
    {
      "role": "user",
      "content": "长沙到乌鲁木齐10月1号高铁和飞机票价对比一下，哪个更方便"
    },
    {
      "role": "assistant",
      "content": "<think>用户想对比高铁和飞机票价，需要分别调用火车票搜索和航班搜索，这两个调用相互独立可以并行。</think>\n\n<tool_call>\n{'name': 'train_tickets_search', 'arguments': '{\"date\": \"2026-10-01\", \"from_city\": \"长沙\", \"to_city\": \"乌鲁木齐\"}'}\n</tool_call>\n<tool_call>\n{'name': 'flights_search', 'arguments': '{\"date\": \"2026-10-01\", \"from_city\": \"长沙\", \"to_city\": \"乌鲁木齐\"}'}\n</tool_call>"
    },
    {
      "role": "user",
      "content": "<tool_response>\n[\"直达车次 Z294，价格 432 元，07:15 从长沙站出发…\"]\n</tool_response>"
    },
    {
      "role": "assistant",
      "content": "<answer>\n长沙到乌鲁木齐10月1日的高铁和飞机对比如下：\n\n**🚄 火车/高铁**：高铁 589 元约 14 小时…\n**✈️ 飞机**：1280-2450 元，约 4.5 小时…\n</answer>"
    }
  ]
}
```

要点：

- **工具结果作为 `role: "user"` 消息注入**，用 `<tool_response>` 包裹（不是 `tool` 角色）。
- 模型可在同一轮**并行发起多个 `<tool_call>`**；信息足够后只输出一次 `<answer>`。
- `<tool_call>` 内容为 Python 风格单引号字典（`{'name': …}`），**并非合法 JSON**，因此解析统一走 `json_repair` 容错处理。

按用途切分为四份：

| 文件 | 用途 | 规模 |
|------|------|------|
| `data/final/sft_train.jsonl` | SFT 训练集 | 850 条 |
| `data/final/sft_val.jsonl` | SFT 验证集 | 31 条 |
| `data/final/rl.jsonl` | GRPO 训练集（同时充当 LLM-Judge 的 gold answer 库） | 200 条 |
| `data/final/test_final.jsonl` | 最终评估测试集 | 80 条 |

---

## 工具系统

8 个工具统一实现为 `qwen_agent.BaseTool`，JSON Schema 定义写在 `src/prompt.py` 的 `COLDSTART_SYSTEM_PROMPT` 中。

| 工具名 | 功能 | 后端 | 实现 |
|--------|------|------|------|
| `search` | 批量网页搜索 | FireCrawl / SerpApi | `tool_web_search.py` / `tool_google_web_search.py` |
| `visit` | 抓取网页并摘要 | FireCrawl / Tavily | `tool_visit.py` / `tool_tavily_visit.py` |
| `weather_search` | 城市天气 | 高德 | `tool_weather.py` |
| `poi_search` | 文本搜索 POI | 高德 | `tool_poi_search.py` |
| `around_search` | 圆心 + 半径搜周边 | 高德 | `tool_around_search.py` |
| `route_planning` | 驾车/步行/骑行/公交路线 | 高德 | `tool_route_planning.py` |
| `flights_search` | 航班查询 | **LLM 模拟** | `tool_transport.py` |
| `train_tickets_search` | 火车票查询 | **LLM 模拟** | `tool_train_ticket.py` |

> `search` 与 `visit` 各有两个实现，**注册了同名工具，只能保留一个**（否则后者覆盖前者）。
> 航班 / 火车票无免费真实 API，由强 LLM 依 `prompt.py` 中的模拟提示词生成符合真实性规则的时刻表与票价；天气 / POI / 路线为高德真实数据。

**输出协议**：

```
<think>...</think>                     ← 可选，推理痕迹
<tool_call>{"name": "...", "arguments": {...}}</tool_call>
<tool_response>...</tool_response>     ← 工具结果
<answer>...</answer>                   ← 最终答案，每轮二选一
```

---

## 数据管线

```bash
python src/data_expansion.py    # ① 种子问题 → 扩充问题（每条改写 5 个）
python src/data_distill.py      # ② 强模型 + 8 真实工具 → 多轮轨迹到 output/
python src/simple_filter.py     # ③ 粗清洗 → data/clean/train.jsonl
python src/format_filter.py     # ④ 标签 → OpenAI tool_calls + 规则过滤
python src/quality_filter.py    # ⑤ LLM 五维打分，保留 pass/borderline
```

质检维度（各 1–5 分，加权后 `overall_score = round(加权和 × 2)`）：
`task_relevance` 0.30 / `completeness` 0.25 / `factual_safety` 0.20 / `tool_use_reasonableness` 0.15 / `format_quality` 0.10。

---

## 训练

### SFT

```bash
bash scripts/train_sft.sh     # LoRA rank 128, 2 epoch, lr 8e-6, max_length 32768, 4 卡
bash scripts/merge_lora.sh    # 合并 LoRA adapter，产出 RL 的初始权重
```

### GRPO

```bash
# 终端 1：启动 vLLM rollout server（2 卡）
bash scripts/rollout.sh

# 终端 2：训练（6 卡）
bash scripts/train_rl.sh
```

关键设计：

- **多轮调度器 `travel_tool_loop`**（`plugins/tooluse_multi_turn_scheduler.py`）
  在 rollout 中真实执行工具，并处理各类异常：未调工具就出答案、畸形 `<tool_call>`、无工具无答案、重复调用死循环（默认连续 3 轮相同则强制收尾）、轮数上限强制收尾。

- **综合奖励 `external_parser_aligned_curriculum_reward`**（`plugins/tooluse_reward_parser_aligned.py`）
  6 个子奖励加权：`process_step` / `tool_schema` / `answer_tag` / `stage_aware` / `tool_efficiency` / `llm_judge`。
  权重按训练进度**课程式**调整（默认三阶段 `[0.15, 0.2, 0.65]`），早期侧重格式与过程，后期 `llm_judge` 占比升至 0.75。

训练脚本中所有超参均可通过环境变量覆盖（`MODEL_PATH`、`RL_DATASET`、`MAX_STEPS`、`LEARNING_RATE`、`MAX_TURNS` 等）。

---

## 评估

```bash
# 端到端多轮工具推理，对 test_final.jsonl 跑全量
MODEL_DIR=output/grpo_parser_aligned_run/<run>/checkpoint-150 bash scripts/infer.sh

# 与 baseline / SFT 做双向 LLM-Judge 对比
python src/llm_judge.py
```

`llm_judge.py` 对同批 query 比较两组轨迹，分**推理路径**（广度 / 需求匹配 / 细节）与**回答结果**（匹配度 / 可行性 / 细节 / 清晰度）两个维度打分，综合分 `combined = 0.3 × Path + 0.7 × Answer`；双向打分（A 左 A 右各一次取平均）以消除位置偏差。

```bash
python src/plot_metrics.py      # 绘制 loss / lr / reward / num_turns 曲线
python src/sample_data.py       # 人工核验任意数据或输出样例
```

---

## 常见问题

**Q：脚本报语法错误？**
训练脚本是 Bash，请在 WSL / Git Bash / Linux 下运行，不要在 PowerShell 或 CMD 中直接执行。

**Q：`search` 工具报错？**
检查 `.env` 中的 `FIRECRAWL_API_KEY`。若改用 SerpApi 版实现，需移除 `tool_web_search.py`（同名工具会互相覆盖）并配置 `SERPAPI_API_KEY`。

**Q：GRPO 时 reward 一直很低？**
查看输出目录下的 `parser_reward_rank{N}.jsonl`，定位是 6 个子奖励中的哪一个拖低。早期 `llm_judge` 权重低是课程学习的正常现象；若 `llm_judge` 恒为 0，检查 `JUDGE_API_KEY` / `JUDGE_BASE_URL` / `JUDGE_MODEL` 是否注入，以及 `data/final/rl.jsonl` 中是否有对应 query 的 gold answer。

**Q：显存不足？**
降低 `MAX_LENGTH` / `MAX_COMPLETION_LENGTH`，开启 `gradient_checkpointing`；SFT 用 LoRA，RL 用 `--deepspeed zero3`。

**Q：模型反复调用同一个工具？**
调度器已内置死循环检测（`TOOL_LOOP_MAX_SAME_TOOL_CALL_ROUNDS`，默认 3）；也可调小 `TOOL_LOOP_FORCE_ANSWER_AFTER_TURNS`（默认 12）让模型更早收尾。

**Q：换更大的模型？**
把各脚本的 `MODEL_PATH` 指向新权重即可（如 Qwen3-14B），同时相应增大显存 / 卡数。

---

## 致谢

训练框架基于 [ms-swift](https://github.com/modelscope/ms-swift)（Apache-2.0）；Agent 工具基类来自 [qwen-agent](https://github.com/QwenLM/Qwen-Agent)；基座模型为 Qwen3。
