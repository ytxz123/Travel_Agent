# -*- coding: utf-8 -*-
# --------------------------------------------
# 文件描述: 多轮工具交互 
# --------------------------------------------

import asyncio
import importlib.util
import json
import os
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from swift.infer_engine.protocol import RolloutOutput
from swift.rollout import MultiTurnScheduler, multi_turns
from swift.utils import remove_response


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


def _load_run_tool_loop_module():
    spec = importlib.util.spec_from_file_location('run_tool_loop_scheduler_bridge', RUN_TOOL_LOOP_INFER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load tool loop helpers from {RUN_TOOL_LOOP_INFER_PATH}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TOOL_LOOP = _load_run_tool_loop_module()

apply_system_tool_call_limit = _TOOL_LOOP.apply_system_tool_call_limit
canonical_tool_call_signature = _TOOL_LOOP.canonical_tool_call_signature
extract_answer_text = _TOOL_LOOP.extract_answer_text
extract_call = _TOOL_LOOP.extract_call
has_answer_start = _TOOL_LOOP.has_answer_start
has_final_answer = _TOOL_LOOP.has_final_answer
has_partial_tool_call = _TOOL_LOOP.has_partial_tool_call
load_tool = _TOOL_LOOP.load_tool
normalize_system_prompt_inplace = _TOOL_LOOP.normalize_system_prompt_inplace
parse_tool_calls = _TOOL_LOOP.parse_tool_calls
retry_route_with_poi_if_needed = _TOOL_LOOP.retry_route_with_poi_if_needed


class _ArgsProxy:

    def __init__(self, tools_dir: str):
        self.tools_dir = tools_dir


class TravelToolLoopScheduler(MultiTurnScheduler):
    """
    Multi-turn scheduler that mirrors `run_tool_loop_infer.py`:
    parse tool calls, execute tools locally, append tool responses, then continue.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tools_dir = os.getenv('TOOL_LOOP_TOOLS_DIR', os.path.join(SRC_DIR, 'tools'))
        self.system_max_tool_calls = int(os.getenv('TOOL_LOOP_SYSTEM_MAX_TOOL_CALLS', '13'))
        self.max_no_tool_no_answer_retries = int(os.getenv('TOOL_LOOP_MAX_NO_TOOL_NO_ANSWER_RETRIES', '2'))
        self.max_invalid_tool_rounds = int(os.getenv('TOOL_LOOP_MAX_INVALID_TOOL_ROUNDS', '3'))
        self.max_same_tool_call_rounds = int(os.getenv('TOOL_LOOP_MAX_SAME_TOOL_CALL_ROUNDS', '3'))
        self.force_answer_after_turns = int(os.getenv('TOOL_LOOP_FORCE_ANSWER_AFTER_TURNS', '12'))
        self.tool_response_max_chars = int(os.getenv('TOOL_LOOP_TOOL_RESPONSE_MAX_CHARS', '5000'))
        self.tool_first_enforce = os.getenv('TOOL_LOOP_TOOL_FIRST_ENFORCE', '1') != '0'
        self.args_proxy = _ArgsProxy(self.tools_dir)

        base_dir = str(Path(self.tools_dir).resolve().parent)
        if base_dir not in os.sys.path:
            os.sys.path.insert(0, base_dir)

    @staticmethod
    def _state_key() -> str:
        return '_travel_tool_loop_state'

    def _initialize_request(self, infer_request) -> Dict[str, Any]:
        state = self._get_state(infer_request)
        if not state['initialized']:
            normalize_system_prompt_inplace(infer_request.messages, self.system_max_tool_calls)
            state['initialized'] = True
        return state

    def _get_state(self, infer_request) -> Dict[str, Any]:
        state = infer_request.data_dict.get(self._state_key())
        if state is None:
            state = {
                'initialized': False,
                'no_tool_no_answer_retries': 0,
                'total_tool_calls': 0,
                'saw_tool_response': False,
                'consecutive_invalid_tool_rounds': 0,
                'previous_tool_call_signature': '',
                'consecutive_same_tool_call_rounds': 0,
                'status': 'running',
                'final_answer': '',
            }
            infer_request.data_dict[self._state_key()] = state
        return state

    @staticmethod
    def _truncate_tool_result(tool_result: Any, max_chars: int) -> str:
        text = str(tool_result)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + '\n...[truncated]'

    @staticmethod
    def _structured_tool_calls_from_choice(response_choice) -> List[Dict[str, Any]]:
        tool_calls = getattr(response_choice.message, 'tool_calls', None) or []
        normalized: List[Dict[str, Any]] = []
        for tool_call in tool_calls:
            if getattr(tool_call, 'function', None):
                tool_call = tool_call["function"]
            name = getattr(tool_call, 'name', None)
            arguments = getattr(tool_call, 'arguments', {})
            if not name:
                continue
            try:
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            except Exception:
                arguments = {'raw_arguments': arguments}
            if not isinstance(arguments, dict):
                arguments = {'raw_arguments': str(arguments)}
            normalized.append({'name': name, 'arguments': arguments})
        return normalized

    def _normalize_response_text(self, response_choice) -> str:
        response_text = response_choice.message.content or ''
        if not response_text and getattr(response_choice.message, 'tool_calls', None):
            response_text = json.dumps(
                self._structured_tool_calls_from_choice(response_choice), ensure_ascii=False)
            response_text = f"<tool_call>\n{response_text}\n</tool_call>"
        return response_text

    def _build_rollout_infos(self, state: Dict[str, Any], response_text: str) -> Dict[str, Any]:
        return {
            'status': state.get('status', 'running'),
            'tool_calls': int(state.get('total_tool_calls', 0)),
            'has_answer_tag': has_final_answer(response_text),
            'final_answer': state.get('final_answer', ''),
        }

    @staticmethod
    def _append_feedback(messages: List[Dict[str, Any]], content: str) -> None:
        # In server rollout mode, ending a turn with `user` breaks template pairing.
        # Use `tool` as the next query role so the following generation remains valid.
        messages.append({'role': 'tool', 'content': f'TOOL_FEEDBACK: {content}'})

    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]], sample_idx: str,
                                  turn: int) -> Dict[str, Any]:
        tool_messages: List[Dict[str, Any]] = []
        valid_tool_calls = 0
        total_tool_calls = 0

        for i, call in enumerate(tool_calls, start=1):
            fn, call_args = extract_call(call)
            tool_call_id = None
            if isinstance(call, dict):
                tool_call_id = call.get('id')
            if not tool_call_id:
                tool_call_id = f'call_{sample_idx}_{turn}_{i}'

            total_tool_calls += 1
            if not fn:
                tool_result = 'TOOL_ERROR: missing function name'
            else:
                valid_tool_calls += 1
                load_error = None
                try:
                    tool = load_tool(self.tools_dir, fn)
                except Exception as e:
                    tool = None
                    load_error = e
                if load_error is not None:
                    tool_result = f'TOOL_ERROR: failed to load tool `{fn}`: {type(load_error).__name__}: {load_error}'
                elif tool is None:
                    tool_result = f'TOOL_ERROR: unsupported tool `{fn}`'
                else:
                    try:
                        tool_result = await tool.call(call_args)
                    except Exception as e:
                        tool_result = f'TOOL_ERROR: {type(e).__name__}: {e}'

                if fn == 'route_planning':
                    retry_result = await retry_route_with_poi_if_needed(self.args_proxy, call_args)
                    if retry_result:
                        tool_result = retry_result

            tool_text = self._truncate_tool_result(tool_result, self.tool_response_max_chars)
            tool_messages.append({
                'role': 'tool',
                'content': tool_text,
                'tool_call_id': tool_call_id,
            })

        return {
            'tool_messages': tool_messages,
            'valid_tool_calls': valid_tool_calls,
            'total_tool_calls': total_tool_calls,
        }

    @staticmethod
    def _run_coro_sync(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result: Dict[str, Any] = {}
        error: Dict[str, BaseException] = {}

        def _runner() -> None:
            try:
                result['value'] = asyncio.run(coro)
            except BaseException as exc:
                error['exc'] = exc

        # `step()` is synchronous, but server rollout calls it from an active event loop.
        # Run tool execution in a dedicated thread so we can safely own a fresh loop there.
        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()

        if 'exc' in error:
            raise error['exc']
        return result['value']

    async def run(self, infer_request, request_config, **kwargs):
        self._initialize_request(infer_request)
        return await super().run(infer_request, request_config, **kwargs)

    def check_finished(self, infer_request, response_choice, current_turn: int) -> bool:
        state = self._get_state(infer_request)
        response_text = self._normalize_response_text(response_choice)

        if response_choice.finish_reason == 'length':
            state['status'] = 'length'
            return True

        if has_final_answer(response_text) or has_answer_start(response_text):
            if not self.tool_first_enforce or state.get('saw_tool_response', False):
                state['status'] = 'answer'
                state['final_answer'] = extract_answer_text(response_text)
                return True

        if self.max_turns and current_turn >= self.max_turns:
            if state.get('status') == 'running':
                state['status'] = 'max_turns'
            return True

        return False

    def step(self, infer_request, response_choice, current_turn: int) -> Dict[str, Any]:
        state = self._initialize_request(infer_request)
        messages = infer_request.messages
        sample_idx = str(getattr(infer_request, 'uuid', 'sample'))

        response_text = self._normalize_response_text(response_choice)
        if messages and messages[-1].get('role') == 'assistant':
            messages[-1]['content'] = response_text

        if (has_final_answer(response_text) or has_answer_start(response_text)) and (
            self.tool_first_enforce and not state.get('saw_tool_response', False)
        ):
            state['status'] = 'answer_before_tool'
            self._append_feedback(messages, '请先通过 <tool_call> 调用至少一个工具，并在收到工具结果后再输出最终<answer>。')
            return {'infer_request': infer_request, 'rollout_infos': self._build_rollout_infos(state, response_text)}

        tool_calls = parse_tool_calls(response_text)
        print("response_text: ", response_text)
        print("parsed_tools: ", tool_calls)
        if not tool_calls and getattr(response_choice.message, 'tool_calls', None):
            tool_calls = self._structured_tool_calls_from_choice(response_choice)

        if not tool_calls and has_partial_tool_call(response_text):
            state['status'] = 'partial_tool_call'
            self._append_feedback(messages, '你上一轮的工具调用不完整。请输出正确的工具调用<tool_call>...</tool_call>。')
            return {'infer_request': infer_request, 'rollout_infos': self._build_rollout_infos(state, response_text)}

        if not tool_calls:
            state['no_tool_no_answer_retries'] += 1
            state['status'] = 'no_tool_no_answer'
            if state.get('saw_tool_response', False) or current_turn >= self.force_answer_after_turns:
                self._append_feedback(messages, '请基于已有信息直接输出最终<answer>...</answer>，不要继续调用工具。')
            else:
                self._append_feedback(
                    messages,
                    '请优先调用最合适的工具来获取事实信息，并仅输出 <tool_call>...</tool_call>'
                    '正确格式示例：<tool_call>\n{"name":"search","arguments":{"query":["杭州西湖门票价格"]}}\n</tool_call>。'
                    )
            return {'infer_request': infer_request, 'rollout_infos': self._build_rollout_infos(state, response_text)}

        current_tool_call_signature = canonical_tool_call_signature(tool_calls)
        if current_tool_call_signature == state.get('previous_tool_call_signature', ''):
            state['consecutive_same_tool_call_rounds'] += 1
        else:
            state['consecutive_same_tool_call_rounds'] = 0
            state['previous_tool_call_signature'] = current_tool_call_signature

        if state['consecutive_same_tool_call_rounds'] >= self.max_same_tool_call_rounds:
            state['status'] = 'repeated_tool_call_loop'
            if state.get('saw_tool_response', False):
                self._append_feedback(messages, '不要重复相同工具调用。请基于已有工具结果直接输出最终<answer>...</answer>。')
            else:
                self._append_feedback(messages, '不要重复相同工具调用。请换一个更合适的工具，输出<tool_call>...</tool_call>。')
            return {'infer_request': infer_request, 'rollout_infos': self._build_rollout_infos(state, response_text)}

        state['no_tool_no_answer_retries'] = 0
        exec_result = self._run_coro_sync(
            self._execute_tool_calls(tool_calls, sample_idx=sample_idx, turn=current_turn))
        state['total_tool_calls'] += exec_result['total_tool_calls']

        if exec_result['valid_tool_calls'] <= 0:
            state['consecutive_invalid_tool_rounds'] += 1
            state['status'] = 'invalid_tool_call_loop'
            self._append_feedback(messages, '你上一轮的工具调用无法执行。请输出正确的工具调用<tool_call>...</tool_call>')
            return {'infer_request': infer_request, 'rollout_infos': self._build_rollout_infos(state, response_text)}

        state['consecutive_invalid_tool_rounds'] = 0
        state['saw_tool_response'] = True
        state['status'] = 'tool_executed'
        messages.extend(exec_result['tool_messages'])

        if current_turn >= self.force_answer_after_turns:
            self._append_feedback(messages, '请基于已获得的工具结果直接输出最终<answer>...</answer>，不要再继续调用工具。')

        return {'infer_request': infer_request, 'rollout_infos': self._build_rollout_infos(state, response_text)}


multi_turns['travel_tool_loop'] = TravelToolLoopScheduler
