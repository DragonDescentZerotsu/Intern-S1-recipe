# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence

import partial_json_parser
from partial_json_parser.core.options import Allow

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger

# TokenizerLike import varies by vLLM version
try:
    from vllm.tokenizers import TokenizerLike
except ModuleNotFoundError:
    # vLLM 0.11.x - use typing.Any as fallback
    from typing import Any as TokenizerLike

try:
    # vLLM 0.9+ / 0.12.0 这类版本
    from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
        ToolParser, ToolParserManager
    )
except ModuleNotFoundError:
    # 新版本兜底 0.13.0+（如果你环境里真有这个路径）
    from vllm.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager

try:
    # vLLM 0.9+ / 0.12.0 这类版本
    from vllm.entrypoints.openai.tool_parsers.utils import extract_intermediate_diff
except ModuleNotFoundError:
    # 新版本兜底 0.13.0+（如果你环境里真有这个路径）
    from vllm.tool_parsers.utils import extract_intermediate_diff

logger = init_logger(__name__)

@ToolParserManager.register_module(["interns1"])
class InternS1ToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        self.position = 0

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        request = super().adjust_request(request)
        if request.tools and request.tool_choice != "none":
            # do not skip special tokens because internlm use the special
            # tokens to indicate the start and end of the tool calls
            # information.
            request.skip_special_tokens = False
        return request

    def get_arguments(self, obj):
        if "parameters" in obj:
            return obj.get("parameters")
        elif "arguments" in obj:
            return obj.get("arguments")
        return None

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        if "<|action_start|>" not in current_text:
            self.position = len(current_text)
            return DeltaMessage(content=delta_text)
        # if the tool call is sent, return an empty delta message
        # to make sure the finish_reason will be sent correctly.
        if self.current_tool_id > 0:
            return DeltaMessage(content="")

        last_pos = self.position
        if "<|action_start|><|plugin|>" not in current_text[last_pos:]:
            return None

        new_delta = current_text[last_pos:]
        text, action = new_delta.split("<|action_start|><|plugin|>")

        if len(text) > 0:
            self.position = self.position + len(text)
            return DeltaMessage(content=text)

        action = action.strip()
        action = action.split("<|action_end|>".strip())[0]

        # bit mask flags for partial JSON parsing. If the name hasn't been
        # sent yet, don't allow sending
        # an incomplete string since OpenAI only ever (as far as I have
        # seen) allows sending the entire tool/ function name at once.
        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR

        try:
            parsable_arr = action

            # tool calls are generated in an object in internlm2
            # it's not support parallel tool calls
            try:
                tool_call_arr: dict = partial_json_parser.loads(parsable_arr, flags)
            except partial_json_parser.core.exceptions.MalformedJSON:
                logger.debug("not enough tokens to parse into JSON yet")
                return None

            # if the current tool name hasn't been sent, send if available
            # - otherwise send nothing
            if not self.current_tool_name_sent:
                function_name = tool_call_arr.get("name")
                if function_name:
                    self.current_tool_id = self.current_tool_id + 1
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=make_tool_call_id(),
                                function=DeltaFunctionCall(
                                    name=function_name
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.current_tool_name_sent = True
                    self.streamed_args_for_tool.append("")
                else:
                    delta = None
            # now we know we're on the same tool call and we're streaming
            # arguments
            else:
                prev_arguments = self.get_arguments(
                    self.prev_tool_call_arr[self.current_tool_id]
                )
                cur_arguments = self.get_arguments(tool_call_arr)

                # not arguments generated
                if not cur_arguments and not prev_arguments:
                    delta = None
                # will never happen
                elif not cur_arguments and prev_arguments:
                    logger.error(
                        "INVARIANT - impossible to have arguments reset mid-arguments"
                    )
                    delta = None
                # first time to get parameters
                elif cur_arguments and not prev_arguments:
                    cur_arguments_json = json.dumps(cur_arguments, ensure_ascii=False)

                    arguments_delta = cur_arguments_json[
                        : cur_arguments_json.index(delta_text) + len(delta_text)
                    ]
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(
                                    arguments=arguments_delta
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += arguments_delta
                # both prev and cur parameters, send the increase parameters
                elif cur_arguments and prev_arguments:
                    cur_args_json = json.dumps(cur_arguments, ensure_ascii=False)
                    prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)

                    argument_diff = extract_intermediate_diff(
                        cur_args_json, prev_args_json
                    )

                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(
                                    arguments=argument_diff
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += argument_diff

            # check to see if the name is defined and has been sent. if so,
            # stream the name - otherwise keep waiting
            # finish by setting old and returning None as base case
            tool_call_arr["arguments"] = self.get_arguments(tool_call_arr)
            self.prev_tool_call_arr = [tool_call_arr]
            return delta
        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            logger.debug(
                "Skipping chunk as a result of tool streaming extraction error"
            )
            return None

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        text = model_output
        tools = request.tools

        START = "<|action_start|><|plugin|>"
        END = "<|action_end|>"

        tool_calls: list[ToolCall] = []
        content_parts: list[str] = []

        pos = 0
        while True:
            s = text.find(START, pos)
            if s == -1:
                content_parts.append(text[pos:])
                break

            # 外部文本
            content_parts.append(text[pos:s])

            e = text.find(END, s + len(START))
            if e == -1:
                # 不完整的 tool block：保守起见当普通文本
                content_parts.append(text[s:])
                break

            action = text[s + len(START): e].strip()
            # 有些输出可能在 JSON 前夹杂换行/多余字符，跟你原逻辑一致：从第一个 "{" 开始截
            j = action.find("{")
            if j != -1:
                action = action[j:]

            try:
                action_dict = json.loads(action)
            except Exception:
                # JSON 解析失败：保守起见当普通文本
                content_parts.append(text[s:e + len(END)])
                pos = e + len(END)
                continue

            name = action_dict.get("name")
            args_obj = action_dict.get("parameters", action_dict.get("arguments", {}))
            args_str = json.dumps(args_obj, ensure_ascii=False)

            if name:
                tool_calls.append(ToolCall(function=FunctionCall(name=name, arguments=args_str)))

            pos = e + len(END)

        content = "".join(content_parts)
        content = content if content.strip() else None

        # 校验工具名是否在 request.tools 中（你原版这里漏了 return）
        if tools:
            allowed = {t.function.name for t in tools}
            tool_calls = [tc for tc in tool_calls if tc.function.name in allowed]

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content,
            )

        return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=text)
