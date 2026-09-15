# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: 自定义reward函数
# --------------------------------------------

import importlib.util
import json
import os
import re
import json_repair
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib import error, request

from swift.rewards import ORM, orms


def _default_project_root() -> str:
    # 项目根目录：优先读取环境变量 TRAVEL_AGENTIC_RL_ROOT；
    # 否则按「本插件位于 <根>/plugins/ 下」向上回溯 1 层推导。
    env_root = os.getenv('TRAVEL_AGENTIC_RL_ROOT', '')
    if env_root:
        return env_root.rstrip('/\\')
    return str(Path(__file__).resolve().parents[1])


PROJECT_ROOT = _default_project_root()
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
RUN_TOOL_LOOP_INFER_PATH = os.path.join(SRC_DIR, 'run_tool_loop_infer.py')
ANSWER_JUDGE_GOLD_DATASET_PATH = os.getenv(
    'ANSWER_JUDGE_GOLD_DATASET_PATH',
)


def _load_run_tool_loop_module():
    spec = importlib.util.spec_from_file_location('run_tool_loop_infer_reward_bridge', RUN_TOOL_LOOP_INFER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load parser from {RUN_TOOL_LOOP_INFER_PATH}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_RUN_TOOL_LOOP_MODULE = _load_run_tool_loop_module()
parse_tool_calls = _RUN_TOOL_LOOP_MODULE.parse_tool_calls


def _extract_first_tag_block(text: str, tag: str) -> Optional[str]:
    pattern = rf'<{tag}>\s*(.*?)\s*</{tag}>'
    m = re.search(pattern, text or '', re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()

def _answer_has_extra_content(text: str) -> bool:
    all_tags = [
        r'<think>.*?</think>',
        r'<answer>.*?</answer>',
        r'<tool_call>.*?</tool_call>',
        r'<tool_response>.*?</tool_response>',
    ]
    for tag in all_tags:
        text = re.sub(tag, '', text, flags=re.DOTALL)
    if text.strip():
        return True
    else:
        return False

def _has_tool_response_in_prompt(prompt: str) -> bool:
    prompt = prompt or ''
    return (
        '<tool_response>' in prompt
        or '<|im_start|>tool' in prompt
        or '"role": "tool"' in prompt
        or "'role': 'tool'" in prompt
    )


def _extract_allowed_tool_names(prompt: str) -> Set[str]:
    tools_block = _extract_first_tag_block(prompt or '', 'tools')
    if not tools_block:
        return set()

    names: Set[str] = set()
    for line in tools_block.splitlines():
        line = line.strip()
        if not line or not line.startswith('{'):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        fn = obj.get('function') if isinstance(obj, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get('name'), str):
            names.add(fn['name'])
    if names:
        return names
    return set(re.findall(r'"name"\s*:\s*"([^"]+)"', tools_block))


def _has_structured_tool_signal(text: str) -> bool:
    text = text or ''
    return (
        '<tool_call>' in text
        or '<tool_calls>' in text
        or '"tool_calls"' in text
        or '"function"' in text
        or '```json' in text
    )


def _is_canonical_tool_call_payload(text: str) -> bool:
    text = (text or '').strip()
    if not text or '```' in text or 'tool_calls' in text or '<tool_calls>' in text:
        return False
        
    tool_calls = parse_tool_calls(text)
        
    if not isinstance(tool_calls, list) or not tool_calls:
        return False
        
    for call in tool_calls:
        if not isinstance(call, dict):
            return False
        if 'function' in call or 'type' in call:
            return False
        extra_keys = set(call.keys()) - {'name', 'arguments'}
        if extra_keys:
            return False
        if not isinstance(call.get('name'), str) or not isinstance(call.get('arguments'), dict):
            return False
    return True


def _clip(v: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, v))


def _extract_user_query_from_prompt(prompt: str) -> str:
    matches = re.findall(r'<\|im_start\|>user\s*(.*?)\s*<\|im_end\|>', prompt or '', re.DOTALL)
    if matches:
        return matches[-1].strip()
    return (prompt or '')[-1200:].strip()


def _extract_final_answer(text: str) -> str:
    answer = _extract_first_tag_block(text or '', 'answer')
    return answer if answer is not None else (text or '').strip()


def _extract_user_query_from_messages(messages: List[Dict[str, Any]]) -> str:
    if not isinstance(messages, list):
        return ''
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get('role') != 'user':
            continue
        content = str(message.get('content', '')).strip()
        if content and '<tool_response>' not in content:
            return content
    return ''


def _extract_allowed_tool_names_from_messages(messages: List[Dict[str, Any]]) -> Set[str]:
    names: Set[str] = set()
    if not isinstance(messages, list):
        return names
    for message in messages:
        if not isinstance(message, dict) or message.get('role') != 'system':
            continue
        names.update(_extract_allowed_tool_names(str(message.get('content', ''))))
    return names


def _resolve_reward_messages(
    sample_messages: Optional[List[Dict[str, Any]]],
    request_id: Optional[str],
    trajectory_inputs: Optional[Dict[str, List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    if request_id and isinstance(trajectory_inputs, dict):
        trajectory = trajectory_inputs.get(request_id) or []
        if isinstance(trajectory, list):
            for item in reversed(trajectory):
                if not isinstance(item, dict):
                    continue
                messages = item.get('messages')
                if isinstance(messages, list) and messages:
                    return messages
    return sample_messages if isinstance(sample_messages, list) else []


def _analyze_assistant_turn(text: str) -> Dict[str, Any]:
    parsed_calls = parse_tool_calls(text)
    answer = _extract_first_tag_block(text, 'answer')
    has_tool_signal = _has_structured_tool_signal(text)
    has_tool_call = len(parsed_calls) > 0
    has_extra_msg = _answer_has_extra_content(text)
    
    return {
        'text': text,
        'parsed_calls': parsed_calls,
        'has_tool_signal': has_tool_signal,
        'has_tool_call': has_tool_call,
        'canonical_payload': _is_canonical_tool_call_payload(text) if has_tool_call else False,
        'has_answer': answer is not None,
        'answer': answer or '',
        'malformed_signal': has_tool_signal and not has_tool_call,
        'has_extra_msg': has_extra_msg
    }


def _collect_assistant_turns(messages: List[Dict[str, Any]], fallback_completion: str) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        turns.append(_analyze_assistant_turn(str(message.get('content', ''))))
    if not turns:
        turns.append(_analyze_assistant_turn(fallback_completion or ''))
    return turns


def _tool_call_signature(call: Dict[str, Any]) -> str:
    return json.dumps(
        {'name': call.get('name'), 'arguments': call.get('arguments', {})},
        sort_keys=True,
        ensure_ascii=False,
    )


def _summarize_intermediate_turns(intermediate_turns: List[Dict[str, Any]], allowed_tools: Set[str]) -> Dict[str, Any]:
    total_calls = 0
    valid_name_count = 0
    parseable_arg_count = 0
    tool_turn_count = 0
    canonical_tool_turn_count = 0
    noncanonical_tool_turn_count = 0
    malformed_turn_count = 0
    structured_tool_turn_count = 0
    intermediate_answer_turn_count = 0
    mixed_turn_count = 0
    idle_turn_count = 0
    has_extra_msg_count = 0
    signatures: List[str] = []

    for turn in intermediate_turns:
        has_signal = turn['has_tool_signal'] or turn['has_tool_call']
        if has_signal:
            structured_tool_turn_count += 1
        if turn['malformed_signal']:
            malformed_turn_count += 1
        if turn['has_answer']:
            intermediate_answer_turn_count += 1
        if turn['has_extra_msg']:
            has_extra_msg_count += 1
        if turn['has_tool_call'] and turn['has_answer']:
            mixed_turn_count += 1
        if turn['has_tool_call']:
            tool_turn_count += 1
            if turn['canonical_payload']:
                canonical_tool_turn_count += 1
            else:
                noncanonical_tool_turn_count += 1
            for call in turn['parsed_calls']:
                total_calls += 1
                if not allowed_tools or call.get('name') in allowed_tools:
                    valid_name_count += 1
                if isinstance(call.get('arguments'), dict):
                    parseable_arg_count += 1
                signatures.append(_tool_call_signature(call))
        elif not turn['has_answer'] and not turn['malformed_signal'] and not (turn['text'] or '').strip():
            idle_turn_count += 1

    print(
            {
            'intermediate_turn_count': len(intermediate_turns),
            'total_calls': total_calls,
            'tool_turn_count': tool_turn_count,
            'has_extra_msg_count': has_extra_msg_count,
            'canonical_tool_turn_count': canonical_tool_turn_count,
            'noncanonical_tool_turn_count': noncanonical_tool_turn_count,
            'structured_tool_turn_count': structured_tool_turn_count,
            'malformed_turn_count': malformed_turn_count,
            'intermediate_answer_turn_count': intermediate_answer_turn_count,
            'mixed_turn_count': mixed_turn_count,
            'idle_turn_count': idle_turn_count,
            'valid_name_count': valid_name_count,
            'parseable_arg_count': parseable_arg_count,
            'unique_signature_count': len(set(signatures)),
            'signatures': signatures,
        }
    )

    return {
        'intermediate_turn_count': len(intermediate_turns),
        'total_calls': total_calls,
        'tool_turn_count': tool_turn_count,
        'canonical_tool_turn_count': canonical_tool_turn_count,
        'noncanonical_tool_turn_count': noncanonical_tool_turn_count,
        'structured_tool_turn_count': structured_tool_turn_count,
        'malformed_turn_count': malformed_turn_count,
        'intermediate_answer_turn_count': intermediate_answer_turn_count,
        'mixed_turn_count': mixed_turn_count,
        'idle_turn_count': idle_turn_count,
        'valid_name_count': valid_name_count,
        'parseable_arg_count': parseable_arg_count,
        'unique_signature_count': len(set(signatures)),
        'signatures': signatures,
    }


def _build_reward_contexts(completions, prompts=None, messages=None, request_id=None, trajectory_inputs=None) -> List[Dict[str, Any]]:
    prompts = prompts or [''] * len(completions)
    contexts: List[Dict[str, Any]] = []
    for i, completion in enumerate(completions):
        prompt = prompts[i] if i < len(prompts) else ''
        sample_messages = None
        if isinstance(messages, list) and i < len(messages) and isinstance(messages[i], list):
            sample_messages = messages[i]
        sample_request_id = None
        if isinstance(request_id, list) and i < len(request_id):
            sample_request_id = request_id[i]
        resolved_messages = _resolve_reward_messages(sample_messages, sample_request_id, trajectory_inputs)
        assistant_turns = _collect_assistant_turns(resolved_messages, completion or '')
        final_turn = assistant_turns[-1]
        intermediate_turns = assistant_turns[:-1]
        allowed_tools = _extract_allowed_tool_names_from_messages(resolved_messages)
        if not allowed_tools:
            allowed_tools = _extract_allowed_tool_names(prompt or '')
        user_query = _extract_user_query_from_messages(resolved_messages)
        if not user_query:
            user_query = _extract_user_query_from_prompt(prompt or '')
        contexts.append({
            'prompt': prompt or '',
            'messages': resolved_messages,
            'request_id': sample_request_id,
            'assistant_turns': assistant_turns,
            'intermediate_turns': intermediate_turns,
            'final_turn': final_turn,
            'allowed_tools': allowed_tools,
            'user_query': user_query,
            'summary': _summarize_intermediate_turns(intermediate_turns, allowed_tools),
        })
    return contexts


def _get_reward_contexts(completions, prompts=None, messages=None, **kwargs) -> List[Dict[str, Any]]:
    contexts = kwargs.get('_parser_reward_contexts')
    if isinstance(contexts, list) and len(contexts) == len(completions):
        return contexts
    return _build_reward_contexts(
        completions,
        prompts=prompts,
        messages=messages or kwargs.get('messages'),
        request_id=kwargs.get('request_id'),
        trajectory_inputs=kwargs.get('trajectory_inputs'),
    )


def _load_gold_answer_by_query(path: str) -> Dict[str, str]:
    gold_answers: Dict[str, str] = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                conversations = row.get('conversations') or row.get('messages') or []
                if not isinstance(conversations, list):
                    continue
                user_query = ''
                gold_answer = ''
                for message in conversations:
                    if not isinstance(message, dict):
                        continue
                    role = message.get('role')
                    content = str(message.get('content', ''))
                    if not user_query and role == 'user' and "<tool_response>" not in content:
                        user_query = content
                    if role == 'assistant':
                        extracted = ""
                        last_open = content.rfind('<answer>')
                        if last_open != -1:
                            last_close = content.rfind('</answer>')
                            if last_close > last_open:
                                extracted = content[last_open + 8:last_close]
                        if extracted:
                            gold_answer = extracted.strip()
                        else:
                            extracted = re.findall(r'<answer>\s*(.*?)\s*</answer>', content, re.DOTALL)
                            if extracted:
                                gold_answer = extracted[-1].strip()
                if user_query and gold_answer:
                    gold_answers[user_query.strip()] = gold_answer
    except Exception:
        return {}
        
    return gold_answers


class ParserAlignedAnswerLLMJudgeReward(ORM):

    ANSWER_ONLY_REWARD_SYSTEM_PROMPT = """你是一名深谙旅游行业、具有严谨逻辑与评测方法论的「旅行规划 LLM Agent 综合评审员」。现需在同一用户 Query 下，参考标准答案，仅对 Agent 的最终回答进行分维度量化评估。请遵循客观公正的原则进行打分，并严格遵循下列指标、打分规则与输出格式。

一、评估内容格式

——————————
<USER_QUERY>
__USER_QUERY__
</USER_QUERY>

<PREDICTION_ANSWER>
__PREDICTION_ANSWER__
</PREDICTION_ANSWER>

<GOLD_ANSWER>
__GOLD_ANSWER__
</GOLD_ANSWER>
——————————

二、回答结果评测（Answer Evaluation）

——————————
【评估维度说明】
1. 匹配度（Relevance）：是否完整响应用户所有子需求、限制和场景要求。
2. 可行性（Feasibility）：方案是否逻辑自洽、切实可行，避免明显冲突或自我矛盾。
3. 细节丰富度（Details）：时间、票价、交通、预算、预约规则、实用提示等信息是否充分且有帮助。
4. 清晰度（Clarity）：结构是否清晰、层次是否分明、表达是否易读。

【评分规则】
• 只评估最终回答，不评估推理路径和工具调用过程。
• 每个维度 0–10 分；0 表示“完全缺失”，8 分以上是“优秀”，10 表示“极为出色”。
• 回答结果综合得分（Overall_A）＝四个维度均值后四舍五入取整。

【注意事项】
• 参考答案和Agent输出答案所使用的日期可能不同，天气、票价、交通费用等具体数字允许有时间差异，因此不要将数字微小差异作为主要扣分依据。
• 所有判断仅基于提供的文本，不要引入外部信息。
——————————

三、综合得分

——————————
综合得分 combined_scores = Overall_A，四舍五入保留 1 位小数。
——————————

【输出格式（严格遵循，不要添加多余内容）】
{
  "analysis": {
    "answer": "<100 字内中文评述：指出Agent答案亮点与不足>"
  },
  "answer_scores": {
    "relevance": <0-10>,
    "feasibility": <0-10>,
    "details": <0-10>,
    "clarity": <0-10>,
    "overall_a": <0-10>
  },
  "combined_scores": <0-10>
}

【重要要求】
• 先逐维度独立思考后再给分，确保公平客观。
• 所有中文评述需具体、可溯源。
• 严格遵守 JSON 模板，以便后续程序解析。

【工具解释】
- visit工具用于访问网页并返回内容摘要。
- search工具用于执行调用google搜索接口，实现通用的、开放知识搜索。
- weather_search工具用于根据城市名称查询指定城市的天气。
- flights_search工具用于根据日期查询从某个城市出发到达某个城市的航班情况。
- train_tickets_search工具用于根据日期查询从某个城市出发到达某个城市的火车票/动车票/高铁票情。
- poi_search工具用于在一个指定的城市内搜索兴趣点（POI）的地理空间信息。
- around_search工具通过设置圆心和半径，搜索圆形区域内的地点信息。
- route_planning工具除提供多种路线规划服务。支持驾车、步行、骑行、电动车、公交路线规划。"""

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        self.api_key = os.getenv('JUDGE_API_KEY') or ''
        self.base_url = os.getenv('JUDGE_BASE_URL', "https://api.zhizengzeng.com/v1").rstrip('/')
        self.model = os.getenv('JUDGE_MODEL') or 'deepseek-v4-flash'
        self.timeout_sec = int(os.getenv('JUDGE_TIMEOUT_SEC', '30'))
        self.gold_answer_by_query = _load_gold_answer_by_query(ANSWER_JUDGE_GOLD_DATASET_PATH)

    @staticmethod
    def _safe_slice(text: str, max_len: int = 5000) -> str:
        text = (text or '').strip()
        if len(text) <= max_len:
            return text
        head = text[: max_len // 2]
        tail = text[-(max_len - len(head)):]
        return f'{head}\n\n...[TRUNCATED]...\n\n{tail}'

    @staticmethod
    def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
        text = (text or '').strip()
        if not text:
            return None
        try:
            val = json.loads(text)
            if isinstance(val, dict):
                return val
        except Exception:
            pass
        fence = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if fence:
            try:
                val = json.loads(fence.group(1))
                if isinstance(val, dict):
                    return val
            except Exception:
                pass
        obj = re.search(r'(\{.*\})', text, re.DOTALL)
        if obj:
            try:
                val = json.loads(obj.group(1))
                if isinstance(val, dict):
                    return val
            except Exception:
                return None
        return None

    @staticmethod
    def _normalize_score(raw_score: Any) -> float:
        try:
            val = float(raw_score)
        except Exception:
            return 0.0
        if val > 1.0:
            val = val / 10.0
        return max(0.0, min(1.0, val))

    def _judge_once(self, user_query: str, prediction_answer: str, gold_answer: str) -> float:
        if not self.api_key or not prediction_answer or not gold_answer:
            return 0.0

        judge_prompt = self.ANSWER_ONLY_REWARD_SYSTEM_PROMPT
        judge_prompt = judge_prompt.replace('__USER_QUERY__', self._safe_slice(user_query, 3000))
        judge_prompt = judge_prompt.replace('__PREDICTION_ANSWER__', self._safe_slice(prediction_answer, 5000))
        judge_prompt = judge_prompt.replace('__GOLD_ANSWER__', self._safe_slice(gold_answer, 5000))

        payload = {
            'model': self.model,
            'messages': [{'role': 'user', 'content': judge_prompt}],
            'temperature': 0,
            'response_format': {'type': 'json_object'},
        }
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        req = request.Request(
            url=f'{self.base_url}/chat/completions',
            data=data,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
            },
            method='POST',
        )
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = json.loads(resp.read().decode('utf-8'))
        except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError):
            return 0.0

        try:
            content = body['choices'][0]['message']['content']
            parsed = self._extract_json_obj(content) if isinstance(content, str) else content
            if not isinstance(parsed, dict):
                return 0.0
            if 'combined_scores' in parsed:
                return self._normalize_score(parsed.get('combined_scores'))
            answer_scores = parsed.get('answer_scores')
            if isinstance(answer_scores, dict) and 'overall_a' in answer_scores:
                return self._normalize_score(answer_scores.get('overall_a'))
            return 0.0
        except Exception:
            return 0.0

    def __call__(self, completions, prompts=None, messages=None, **kwargs) -> List[float]:
        contexts = _get_reward_contexts(completions, prompts=prompts, messages=messages, **kwargs)
        rewards: List[float] = []
        for context in contexts:
            user_query = context['user_query']
            prediction_answer = context['final_turn']['answer'] or _extract_final_answer(context['final_turn']['text'])
            gold_answer = self.gold_answer_by_query.get(user_query.strip(), '')
            print("query: ", user_query)
            print("prediction: ", prediction_answer)
            print("answer: ", gold_answer)
            rewards.append(self._judge_once(user_query, prediction_answer, gold_answer))
        return rewards


orms['external_parser_aligned_llm_judge_reward'] = ParserAlignedAnswerLLMJudgeReward


class ParserAlignedAnswerTagReward(ORM):

    def __call__(self, completions, prompts=None, messages=None, **kwargs) -> List[float]:
        contexts = _get_reward_contexts(completions, prompts=prompts, messages=messages, **kwargs)
        rewards: List[float] = []
        for context in contexts:
            final_turn = context['final_turn']
            answer = final_turn['answer']
            if final_turn['has_tool_call']:
                rewards.append(-1.0)
            elif not answer:
                rewards.append(-1.0)
            elif final_turn["has_extra_msg"]:
                rewards.append(-0.2)
            elif len(answer) < 10:
                rewards.append(0.0)
            else:
                rewards.append(0.3)
        return rewards


orms['external_parser_aligned_answer_tag_reward'] = ParserAlignedAnswerTagReward


class ParserAlignedToolSchemaReward(ORM):

    def __call__(self, completions, prompts=None, messages=None, **kwargs) -> List[float]:
        contexts = _get_reward_contexts(completions, prompts=prompts, messages=messages, **kwargs)
        rewards: List[float] = []
        for context in contexts:
            summary = context['summary']
            total_calls = summary['total_calls']
            if total_calls == 0:
                if summary['malformed_turn_count'] > 0:
                    rewards.append(-1.0)
                else:
                    rewards.append(0.0)
                continue

            tool_turn_count = max(1, summary['tool_turn_count'])
            valid_name_ratio = summary['valid_name_count'] / total_calls
            parseable_ratio = summary['parseable_arg_count'] / total_calls
            canonical_ratio = summary['canonical_tool_turn_count'] / tool_turn_count
            quality = (valid_name_ratio + parseable_ratio + canonical_ratio) / 3.0
            reward = -0.2 + 0.45 * quality
            reward = min(reward, 0.3)
            if summary['noncanonical_tool_turn_count'] > 0:
                reward = min(reward, -0.4)
            if summary['malformed_turn_count'] > 0:
                malformed_ratio = summary['malformed_turn_count'] / max(1, summary['structured_tool_turn_count'])
                reward -= 0.8 * malformed_ratio
            rewards.append(_clip(reward))
        return rewards


orms['external_parser_aligned_tool_schema_reward'] = ParserAlignedToolSchemaReward


class ParserAlignedStageReward(ORM):

    def __call__(self, completions, prompts=None, messages=None, **kwargs) -> List[float]:
        contexts = _get_reward_contexts(completions, prompts=prompts, messages=messages, **kwargs)
        rewards: List[float] = []
        for context in contexts:
            summary = context['summary']
            final_turn = context['final_turn']

            if final_turn['has_tool_call']:
                rewards.append(-1.0)
                continue

            if summary['mixed_turn_count'] > 0:
                rewards.append(-1.0)
                continue

            if summary['noncanonical_tool_turn_count'] > 0:
                rewards.append(-0.4)
                continue

            if summary['intermediate_answer_turn_count'] > 0:
                rewards.append(-0.8)
                continue

            if not final_turn['has_answer']:
                rewards.append(-0.8 if summary['total_calls'] == 0 else -0.5)
                continue

            if summary['total_calls'] == 0:
                rewards.append(-0.1)
                continue

            if summary['malformed_turn_count'] > 0:
                rewards.append(0.0)
                continue

            rewards.append(0.3)
        return rewards


orms['external_parser_aligned_stage_reward'] = ParserAlignedStageReward


class ParserAlignedProcessReward(ORM):

    def __call__(self, completions, prompts=None, messages=None, **kwargs) -> List[float]:
        contexts = _get_reward_contexts(completions, prompts=prompts, messages=messages, **kwargs)
        rewards: List[float] = []
        for context in contexts:
            summary = context['summary']
            final_turn = context['final_turn']

            score = 0.0
            if summary['total_calls'] > 0:
                score += 0.1
                score += min(0.1, 0.03 * summary['total_calls'])
                score += 0.1 * (summary['parseable_arg_count'] / summary['total_calls'])
            else:
                score -= 0.1

            if summary['tool_turn_count'] > 0:
                score += 0.05 * (summary['canonical_tool_turn_count'] / summary['tool_turn_count'])

            score -= 0.6 * summary['noncanonical_tool_turn_count']
            score -= 0.8 * summary['malformed_turn_count']
            score -= 0.5 * summary['intermediate_answer_turn_count']
            score -= 0.4 * summary['mixed_turn_count']
            score -= 0.2 * summary['idle_turn_count']

            if final_turn['has_answer'] and not final_turn['has_tool_call']:
                score += 0.05
            elif final_turn['has_tool_call']:
                score -= 0.5
            else:
                score -= 0.2

            rewards.append(_clip(score, low=-1.0, high=0.3))
        return rewards


orms['external_parser_aligned_process_reward'] = ParserAlignedProcessReward


class ParserAlignedToolEfficiencyReward(ORM):

    @staticmethod
    def _signature(call: Dict[str, Any]) -> str:
        return _tool_call_signature(call)

    def __call__(self, completions, prompts=None, messages=None, **kwargs) -> List[float]:
        contexts = _get_reward_contexts(completions, prompts=prompts, messages=messages, **kwargs)
        rewards: List[float] = []
        for context in contexts:
            summary = context['summary']
            if summary['total_calls'] == 0:
                if summary['malformed_turn_count'] > 0:
                    rewards.append(-0.3)
                else:
                    rewards.append(0.0)
                continue

            signatures = summary['signatures']
            unique_ratio = summary['unique_signature_count'] / max(1, len(signatures))
            duplicate_ratio = 1.0 - unique_ratio
            over_call_penalty = max(0, len(signatures) - 10)
            reward = 0.3 * unique_ratio - 0.3 * duplicate_ratio -  0.03 * over_call_penalty
            rewards.append(_clip(reward, low=-1.0, high=0.3))
        return rewards


orms['external_parser_aligned_efficiency_reward'] = ParserAlignedToolEfficiencyReward


class ParserAlignedCurriculumReward(ORM):

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        self.total_steps = max(1, int(os.getenv('PARSER_REWARD_TOTAL_STEPS', '100')))
        self.phase_ratios = self._normalize_phase_ratios(
            os.getenv('PARSER_REWARD_PHASE_RATIOS', '[0.2, 0.35, 0.45]')
        )
        self.w1 = self._normalize_weights(
            os.getenv('PARSER_REWARD_W1', '[0.10, 0.10, 0.05, 0.05, 0.10, 0.60]')
        )
        self.w2 = self._normalize_weights(
            os.getenv('PARSER_REWARD_W2', '[0.08, 0.08, 0.04, 0.05, 0.10, 0.65]')
        )
        self.w3 = self._normalize_weights(
            os.getenv('PARSER_REWARD_W3', '[0.05, 0.07, 0.03, 0.05, 0.10, 0.70]')
        )

        self._process = ParserAlignedProcessReward(args)
        self._schema = ParserAlignedToolSchemaReward(args)
        self._answer = ParserAlignedAnswerTagReward(args)
        self._stage = ParserAlignedStageReward(args)
        self._efficiency = ParserAlignedToolEfficiencyReward(args)
        self._judge = ParserAlignedAnswerLLMJudgeReward(args)

        rank = str(os.getenv('RANK', '0'))
        base_output_dir = getattr(args, 'output_dir', os.getcwd()) if args is not None else os.getcwd()
        self.debug_enabled = str(os.getenv('PARSER_REWARD_DEBUG', '1')) == '1'
        self.debug_log_path = os.path.join(base_output_dir, f'parser_reward_rank{rank}.jsonl')

    @staticmethod
    def _normalize_phase_ratios(raw: str) -> List[float]:
        try:
            phase_ratios = json.loads(raw)
        except Exception:
            phase_ratios = [0.15, 0.25, 0.6]
        if not isinstance(phase_ratios, list) or len(phase_ratios) not in {2, 3}:
            phase_ratios = [0.15, 0.25, 0.6]
        vals = [max(0.0, float(v)) for v in phase_ratios]
        s = sum(vals) or 1.0
        return [v / s for v in vals]

    @staticmethod
    def _normalize_weights(raw: str) -> List[float]:
        try:
            weights = json.loads(raw)
        except Exception:
            weights = [0.05, 0.05, 0.05, 0.05, 0.10, 0.70]
        if isinstance(weights, list) and len(weights) == 5:
            weights = weights + [0.0]
        if not isinstance(weights, list) or len(weights) != 6:
            weights = [0.05, 0.05, 0.05, 0.05, 0.10, 0.70]
        vals = [max(0.0, float(v)) for v in weights]
        s = sum(vals) or 1.0
        return [v / s for v in vals]

    def _current_weights(self, global_step: int) -> List[float]:
        boundaries: List[int] = []
        cumulative = 0.0
        for ratio in self.phase_ratios[:-1]:
            cumulative += ratio
            boundaries.append(int(self.total_steps * cumulative))
        phases = [self.w1, self.w2, self.w3]
        if len(self.phase_ratios) == 2:
            phases = [self.w1, self.w3]
        for idx, boundary in enumerate(boundaries):
            if global_step < boundary:
                return phases[idx]
        return phases[len(boundaries)]

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    def _write_debug(self, global_step: int, weights: List[float], rewards: Dict[str, List[float]],
                     mixed_rewards: List[float]) -> None:
        if not self.debug_enabled:
            return
        payload = {
            'global_step': int(global_step),
            'weights': [float(x) for x in weights],
            'sub_reward_mean': {k: self._safe_mean(v) for k, v in rewards.items()},
            'mixed_reward_mean': self._safe_mean(mixed_rewards),
        }
        try:
            os.makedirs(os.path.dirname(self.debug_log_path), exist_ok=True)
            with open(self.debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')
        except Exception:
            pass

    def __call__(self, completions, prompts=None, **kwargs) -> List[float]:
        prompts = prompts or kwargs.get('prompts') or [''] * len(completions)
        reward_contexts = _get_reward_contexts(completions, prompts=prompts, **kwargs)
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step if trainer_state is not None else 0
        weights = self._current_weights(global_step)
        reward_kwargs = dict(kwargs)
        reward_kwargs['_parser_reward_contexts'] = reward_contexts

        r_process = self._process(completions, prompts=prompts, **reward_kwargs)
        r_schema = self._schema(completions, prompts=prompts, **reward_kwargs)
        r_answer = self._answer(completions, prompts=prompts, **reward_kwargs)
        r_stage = self._stage(completions, prompts=prompts, **reward_kwargs)
        r_efficiency = self._efficiency(completions, prompts=prompts, **reward_kwargs)
        r_judge = self._judge(completions, prompts=prompts, **reward_kwargs)
        print("reward: ", r_process, r_schema, r_answer, r_stage, r_efficiency, r_judge)

        rewards: List[float] = []
        for i in range(len(completions)):
            mixed = (
                weights[0] * r_process[i]
                + weights[1] * r_schema[i]
                + weights[2] * r_answer[i]
                + weights[3] * r_stage[i]
                + weights[4] * r_efficiency[i]
                + weights[5] * r_judge[i]
            )
            rewards.append(_clip(mixed))

        self._write_debug(
            global_step,
            weights,
            {
                'process_step': r_process,
                'tool_schema': r_schema,
                'answer_tag': r_answer,
                'stage_aware': r_stage,
                'tool_efficiency': r_efficiency,
                'llm_judge': r_judge,
            },
            rewards,
        )
        return rewards


orms['external_parser_aligned_curriculum_reward'] = ParserAlignedCurriculumReward
