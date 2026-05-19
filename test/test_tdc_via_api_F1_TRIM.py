import argparse
import json
import logging
import multiprocessing
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm


load_dotenv(override=True)

current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent
trim_root = Path("/data1/tianang/Projects/TRIM")
trim_src = trim_root / "src"
tools_dir = project_root / "tools"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if trim_src.exists() and str(trim_src) not in sys.path:
    sys.path.insert(0, str(trim_src))
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))

from utils.TDC_answer_parser import extract_answer, parse_answer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


OPENAI_OFFICIAL_API_BASE = "https://api.openai.com/v1"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEFAULT_LOCAL_API_BASE = "http://localhost:8001/v1"
DEFAULT_API_KEY = "EMPTY"
OPENAI_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")
OPENAI_REASONING_SUMMARIES = ("none", "auto", "concise", "detailed")
DEEPSEEK_REASONING_EFFORTS = ("high", "max")
TOOL_MODES = ("none", "properties", "similar", "both", "chembl", "properties_chembl", "both_chembl")
SIMILAR_TOOL_FEATURE_VIEWS = ("all", "properties", "functional_groups", "neighbors_only")
SIMILAR_TOOL_PROPERTY_LINE_COUNTS = (9, 18, 27, 36)
TRIM_TOOL_NAMES_BY_MODE = {
    "none": (),
    "properties": ("get_mol_properties_and_fg",),
    "similar": ("compare_similar_mols",),
    "both": ("get_mol_properties_and_fg", "compare_similar_mols"),
    "chembl": (),
    "properties_chembl": ("get_mol_properties_and_fg",),
    "both_chembl": ("get_mol_properties_and_fg", "compare_similar_mols"),
}
CHEMBL_TOOL_MODES = {"chembl", "properties_chembl", "both_chembl"}
SMILES_IDENTITY_HEURISTIC_GUARD = (
    "Do not infer the label by recognizing or memorizing the molecule's identity or name from the SMILES string; "
    "do not compare it against similar molecules you already know from memory; "
    "base your prediction only on structural decomposition of the molecule and properties obtained from the available tools."
)


_CLIENT = None
_TOOL_RUNTIME = None


def is_openai_official_model(model_name: str) -> bool:
    return bool(model_name) and (model_name.startswith("gpt-") or model_name.startswith("o"))


def is_deepseek_direct_model(model_name: str) -> bool:
    return bool(model_name) and model_name.startswith("deepseek-v4-")


def is_openai_official_api_base(api_base: str) -> bool:
    return api_base.rstrip("/") == OPENAI_OFFICIAL_API_BASE.rstrip("/")


def is_openrouter_api_base(api_base: str) -> bool:
    return api_base.rstrip("/") == OPENROUTER_API_BASE.rstrip("/")


def is_deepseek_api_base(api_base: str) -> bool:
    normalized = api_base.rstrip("/")
    return normalized == DEEPSEEK_API_BASE.rstrip("/") or normalized.startswith(f"{DEEPSEEK_API_BASE.rstrip('/')}/")


def get_args():
    parser = argparse.ArgumentParser(description="Run TRIM TDC F1 eval via OpenAI-compatible APIs.")

    parser.add_argument(
        "--provider",
        choices=["auto", "local", "openai", "openrouter", "deepseek"],
        default="auto",
        help=(
            "API provider. auto infers OpenAI from gpt/o-series model IDs, "
            "DeepSeek from deepseek-v4-* model IDs, and OpenRouter from model IDs containing '/'."
        ),
    )
    parser.add_argument("--api-base", type=str, default=DEFAULT_LOCAL_API_BASE, help="API base URL.")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="API key. Use EMPTY for local serve.")
    parser.add_argument("--model", type=str, default="", help="Model name. If empty, query the API server.")
    parser.add_argument(
        "--openrouter-api-key-env",
        type=str,
        default="OPENROUTER_API_KEY",
        help="Primary environment variable to use for OpenRouter when --api-key is not set.",
    )
    parser.add_argument(
        "--openai-reasoning-effort",
        type=str,
        default="medium",
        choices=OPENAI_REASONING_EFFORTS,
        help="Reasoning effort for official OpenAI Chat Completions.",
    )
    parser.add_argument(
        "--openai-reasoning-summary",
        type=str,
        default="none",
        choices=OPENAI_REASONING_SUMMARIES,
        help=(
            "Responses API reasoning summary verbosity for official OpenAI reasoning models. "
            "Use auto, concise, or detailed to save reasoning summary items in trajectories."
        ),
    )
    parser.add_argument(
        "--deepseek-reasoning-effort",
        type=str,
        default="high",
        choices=DEEPSEEK_REASONING_EFFORTS,
        help="Reasoning effort for DeepSeek V4 thinking mode.",
    )
    parser.add_argument("--max-tokens", type=int, default=10240, help="Max generated tokens.")
    parser.add_argument(
        "--chat-template-kwargs-json",
        type=str,
        default="",
        help=(
            "Optional JSON object passed to OpenAI-compatible servers as "
            "extra_body.chat_template_kwargs, e.g. '{\"enable_thinking\": true}' for Gemma 4 on vLLM."
        ),
    )
    parser.add_argument(
        "--openai-input-price-per-mtok",
        type=float,
        default=0.75,
        help="Official OpenAI input price in USD per 1M input tokens.",
    )
    parser.add_argument(
        "--openai-output-price-per-mtok",
        type=float,
        default=4.5,
        help="Official OpenAI output price in USD per 1M output tokens.",
    )
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True, help="Enable provider reasoning when supported.")

    parser.add_argument(
        "--task-groups",
        nargs="+",
        default=["Tox", "ADME", "HTS"],
        choices=["ADME", "Tox", "HTS", "all"],
        help="Task groups to run.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional explicit task names to run, e.g. DILI BBB_Martins.",
    )
    parser.add_argument(
        "--prompt-task",
        type=str,
        default=None,
        help=(
            "Optional TRIM prompt task name to use for all input files. "
            "Use this when evaluating custom jsonl files that should reuse an existing task prompt, e.g. BBB_Martins."
        ),
    )
    parser.add_argument(
        "--disallow-smiles-identity-heuristics",
        action="store_true",
        default=False,
        help=(
            "Append an instruction forbidding predictions based on recognizing the molecule identity/name from SMILES; "
            "the model should use structural decomposition and tool-derived properties only."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "DataPrepare/TDC_no_conflict_labels_salt_removed/valid",
        help="Directory containing TRIM salt-removed TDC jsonl files.",
    )
    parser.add_argument("--n-samples", type=int, default=1, help="Number of model samples per molecule.")
    parser.add_argument("--num-processes", type=int, default=8, help="Number of parallel workers.")
    parser.add_argument("--max-retry", type=int, default=4, help="Max retries when answer parsing fails.")
    parser.add_argument("--limit-samples", type=int, default=None, help="Optional per-task sample limit for smoke tests.")

    parser.add_argument(
        "--tool-mode",
        choices=TOOL_MODES,
        default="none",
        help=(
            "Tools available to the model: none, properties, similar, both, chembl, "
            "properties_chembl, or both_chembl."
        ),
    )
    parser.add_argument(
        "--first-turn-tool-choice",
        choices=["auto", "required"],
        default="auto",
        help=(
            "Tool choice for the first Chat Completions turn when tools are enabled. "
            "Use required for smoke tests that must verify the tool round-trip."
        ),
    )
    parser.add_argument(
        "--chembl-include-activities",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Hidden backend setting for chembl_info. The model still only sees the same "
            "chembl_info(smiles) schema; use --no-chembl-include-activities for a "
            "properties-only control."
        ),
    )
    parser.add_argument(
        "--debug-parse-failures",
        action="store_true",
        default=False,
        help="Log a short raw response excerpt when answer parsing fails.",
    )
    parser.add_argument(
        "--neighbors-per-label",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Neighbors per label for compare_similar_mols.",
    )
    parser.add_argument(
        "--similar-tool-feature-view",
        choices=SIMILAR_TOOL_FEATURE_VIEWS,
        default="all",
        help=(
            "Postprocess compare_similar_mols output without changing the TRIM tool runtime: "
            "all keeps the original text, properties drops functional-group differences, "
            "functional_groups drops property comparisons, neighbors_only drops both."
        ),
    )
    parser.add_argument(
        "--similar-tool-property-lines",
        type=int,
        default=None,
        choices=SIMILAR_TOOL_PROPERTY_LINE_COUNTS,
        help=(
            "Optional number of property comparison lines to keep per neighbor in compare_similar_mols. "
            "Use with --similar-tool-feature-view properties for properties-only ablations; 36 is the full current property set."
        ),
    )
    parser.add_argument("--log-file", action="store_true", default=False, help="Save logs to file.")
    parser.add_argument("--log-file-name", type=str, default="trim_api_f1_{t_stamp}.log", help="Log file name.")
    parser.add_argument(
        "--save-reasoning-trajectories",
        action="store_true",
        default=False,
        help=(
            "Save one JSONL trajectory record per evaluated sample under "
            "reasoning-trajectory/<formatted log-file-name>/<task>.jsonl."
        ),
    )

    return parser.parse_args()


def infer_provider(args) -> str:
    if args.provider != "auto":
        return args.provider
    if is_openai_official_api_base(args.api_base):
        return "openai"
    if is_deepseek_api_base(args.api_base):
        return "deepseek"
    if is_openrouter_api_base(args.api_base):
        return "openrouter"
    if is_openai_official_model(args.model):
        return "openai"
    if is_deepseek_direct_model(args.model):
        return "deepseek"
    if "/" in args.model:
        return "openrouter"
    return "local"


def resolve_api_settings(args):
    provider = infer_provider(args)
    args.provider = provider
    args.chat_template_kwargs = parse_json_object_arg(
        args.chat_template_kwargs_json,
        "--chat-template-kwargs-json",
    )

    if provider == "openai":
        if args.api_base == DEFAULT_LOCAL_API_BASE:
            args.api_base = OPENAI_OFFICIAL_API_BASE
        if args.api_key == DEFAULT_API_KEY:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required in .env or environment for --provider openai.")
            args.api_key = api_key
    elif provider == "deepseek":
        if args.api_base == DEFAULT_LOCAL_API_BASE:
            args.api_base = DEEPSEEK_API_BASE
        if not args.model:
            args.model = "deepseek-v4-pro"
        if args.api_key == DEFAULT_API_KEY:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY is required in .env or environment for --provider deepseek.")
            args.api_key = api_key
    elif provider == "openrouter":
        if args.api_base == DEFAULT_LOCAL_API_BASE:
            args.api_base = OPENROUTER_API_BASE
        if args.api_key == DEFAULT_API_KEY:
            api_key = first_env_value(
                args.openrouter_api_key_env,
                "OPENROUTER_API_KEY",
                "OPENROUTER_API_KEY_Mark_1",
                "OPENROUTER_API_KEY_Mark",
                "OPENROUTER_API_KEY_Haydn",
            )
            if not api_key:
                raise ValueError(
                    f"{args.openrouter_api_key_env} or a known OPENROUTER_API_KEY_* variable "
                    "is required for --provider openrouter."
                )
            args.api_key = api_key

    return args


def parse_json_object_arg(value: str, arg_name: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{arg_name} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{arg_name} must decode to a JSON object.")
    return parsed


def first_env_value(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def read_usage_value(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_usage_cost(response, *, provider: str, args) -> dict[str, float | int | None]:
    usage = getattr(response, "usage", None)
    prompt_tokens = read_usage_value(usage, "prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = read_usage_value(usage, "input_tokens")
    completion_tokens = read_usage_value(usage, "completion_tokens")
    if completion_tokens is None:
        completion_tokens = read_usage_value(usage, "output_tokens")
    total_tokens = read_usage_value(usage, "total_tokens")
    completion_details = read_usage_value(usage, "completion_tokens_details", {}) or {}
    if not completion_details:
        completion_details = read_usage_value(usage, "output_tokens_details", {}) or {}
    reasoning_tokens = read_usage_value(completion_details, "reasoning_tokens")

    reported_cost = read_usage_value(usage, "cost")
    estimated_cost = None
    if provider == "openai" and prompt_tokens is not None and completion_tokens is not None:
        estimated_cost = (
            (int(prompt_tokens) / 1_000_000.0) * float(args.openai_input_price_per_mtok)
            + (int(completion_tokens) / 1_000_000.0) * float(args.openai_output_price_per_mtok)
        )

    cost = reported_cost if reported_cost is not None else estimated_cost
    try:
        cost = float(cost) if cost is not None else None
    except (TypeError, ValueError):
        cost = None

    return {
        "prompt_tokens": int(prompt_tokens) if prompt_tokens is not None else 0,
        "completion_tokens": int(completion_tokens) if completion_tokens is not None else 0,
        "total_tokens": int(total_tokens) if total_tokens is not None else 0,
        "reasoning_tokens": int(reasoning_tokens) if reasoning_tokens is not None else 0,
        "cost_usd": cost,
    }


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True, mode="json")
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def normalize_messages_for_trace(messages: list[Any]) -> list[dict[str, Any]]:
    normalized = []
    for message in messages:
        item = to_jsonable(message)
        if isinstance(item, Mapping):
            item = dict(item)
            if "reasoning" not in item and isinstance(item.get("reasoning_content"), str):
                item["reasoning"] = item["reasoning_content"]
            normalized.append(item)
        else:
            normalized.append({"value": item})
    return normalized


def extract_reasoning_summaries(trace_messages: list[dict[str, Any]]) -> list[str]:
    summaries: list[str] = []
    for message in trace_messages:
        if message.get("type") != "reasoning":
            continue
        for item in message.get("summary") or []:
            if isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    summaries.append(text.strip())
    return summaries


def to_chat_completion_tool_schema(schema: dict[str, object]) -> dict[str, object]:
    """TRIM exposes flat tool schemas; Chat Completions expects function nesting."""
    if "function" in schema:
        return schema
    return {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema.get("description", ""),
            "parameters": schema["parameters"],
            "strict": bool(schema.get("strict", True)),
        },
    }


def to_responses_tool_schema(schema: dict[str, object]) -> dict[str, object]:
    """Responses API expects the flat TRIM/OpenAI tool schema."""
    if "function" not in schema:
        return schema
    function_schema = schema["function"]
    return {
        "type": "function",
        "name": function_schema["name"],
        "description": function_schema.get("description", ""),
        "parameters": function_schema["parameters"],
        "strict": bool(function_schema.get("strict", True)),
    }


def get_trim_tools_for_mode(tool_mode: str, *, api_style: str = "chat") -> list[dict[str, object]] | None:
    if tool_mode == "none":
        return None

    global _TOOL_RUNTIME
    allowed_names = set(TRIM_TOOL_NAMES_BY_MODE[tool_mode])
    tools = []
    if allowed_names:
        from trim.reasoning.agent_tools import build_openai_tool_runtime

        if _TOOL_RUNTIME is None:
            _TOOL_RUNTIME = build_openai_tool_runtime()

        for schema in _TOOL_RUNTIME.tools:
            name = str(schema.get("name") or schema.get("function", {}).get("name"))
            if name in allowed_names:
                if api_style == "responses":
                    tools.append(to_responses_tool_schema(schema))
                else:
                    tools.append(to_chat_completion_tool_schema(schema))

    if tool_mode in CHEMBL_TOOL_MODES:
        from chembl_info import CHEMBL_INFO_OPENAI_TOOLS

        for schema in CHEMBL_INFO_OPENAI_TOOLS:
            if api_style == "responses":
                tools.append(to_responses_tool_schema(schema))
            else:
                tools.append(to_chat_completion_tool_schema(schema))
    return tools


def init_worker(api_base, api_key, provider, tool_mode):
    global _CLIENT, _TOOL_RUNTIME
    _CLIENT = OpenAI(
        api_key=api_key,
        base_url=api_base,
        max_retries=1,
        timeout=600.0 if provider == "openai" else 180.0,
    )
    if TRIM_TOOL_NAMES_BY_MODE[tool_mode]:
        from trim.reasoning.agent_tools import build_openai_tool_runtime

        _TOOL_RUNTIME = build_openai_tool_runtime()


def message_role(message) -> str | None:
    if isinstance(message, Mapping):
        role = message.get("role")
    else:
        role = getattr(message, "role", None)
    return str(role) if role is not None else None


def create_chat_completion(client, *, provider, model_name, messages, tools, args):
    request_kwargs = {
        "model": model_name,
        "messages": messages,
    }
    if tools is not None:
        request_kwargs["tools"] = tools
        if (
            getattr(args, "first_turn_tool_choice", "auto") == "required"
            and not any(message_role(message) == "tool" for message in messages)
        ):
            request_kwargs["tool_choice"] = "required"

    if provider == "openai":
        request_kwargs["max_completion_tokens"] = args.max_tokens
        # OpenAI Chat Completions currently rejects gpt-5.4-mini requests that
        # combine function tools with reasoning_effort. Keep reasoning_effort
        # for no-tool OpenAI runs, and rely on the prompt for visible rationale
        # when tools are enabled.
        if tools is None:
            request_kwargs["reasoning_effort"] = args.openai_reasoning_effort
    else:
        request_kwargs["max_tokens"] = args.max_tokens
        extra_body = {}
        if args.chat_template_kwargs:
            extra_body["chat_template_kwargs"] = args.chat_template_kwargs
        if provider == "openrouter":
            extra_body["usage"] = {"include": True}
            if args.thinking:
                extra_body["reasoning"] = {"enabled": True}
        elif provider == "deepseek":
            extra_body["thinking"] = {"type": "enabled" if args.thinking else "disabled"}
            if args.thinking:
                request_kwargs["reasoning_effort"] = args.deepseek_reasoning_effort
        if extra_body:
            request_kwargs["extra_body"] = extra_body

    return client.chat.completions.create(**request_kwargs)


def create_openai_response(client, *, model_name, input_items, tools, args):
    reasoning = {"effort": args.openai_reasoning_effort}
    if args.openai_reasoning_summary != "none":
        reasoning["summary"] = args.openai_reasoning_summary

    request_kwargs = {
        "model": model_name,
        "input": input_items,
        "tools": tools,
        "reasoning": reasoning,
        "max_output_tokens": args.max_tokens,
    }
    if (
        tools is not None
        and getattr(args, "first_turn_tool_choice", "auto") == "required"
        and not any(item.get("type") == "function_call_output" for item in input_items if isinstance(item, Mapping))
    ):
        request_kwargs["tool_choice"] = "required"
    return client.responses.create(**request_kwargs)


def get_tool_call_name(tool_call) -> str | None:
    if isinstance(tool_call, Mapping):
        function_payload = tool_call.get("function")
        if isinstance(function_payload, Mapping):
            name = function_payload.get("name")
        else:
            name = tool_call.get("name")
    else:
        function_payload = getattr(tool_call, "function", None)
        name = (
            getattr(function_payload, "name", None)
            if function_payload is not None
            else getattr(tool_call, "name", None)
        )
    return str(name) if name else None


def get_tool_call_arguments(tool_call) -> dict[str, Any]:
    if isinstance(tool_call, Mapping):
        function_payload = tool_call.get("function")
        if isinstance(function_payload, Mapping):
            raw_args = function_payload.get("arguments")
        else:
            raw_args = tool_call.get("arguments")
    else:
        function_payload = getattr(tool_call, "function", None)
        raw_args = (
            getattr(function_payload, "arguments", None)
            if function_payload is not None
            else getattr(tool_call, "arguments", None)
        )
    if raw_args is None:
        return {}
    if isinstance(raw_args, Mapping):
        return dict(raw_args)
    try:
        parsed = json.loads(str(raw_args))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def postprocess_compare_similar_mols_text(
    text: str,
    *,
    feature_view: str,
    property_lines: int | None,
) -> str:
    if feature_view == "all" and property_lines is None:
        return text

    include_properties = feature_view in {"all", "properties"}
    include_functional_groups = feature_view in {"all", "functional_groups"}
    output_lines: list[str] = []
    section: str | None = None
    kept_property_lines = 0

    for line in text.splitlines():
        if line == "properties:":
            section = "properties"
            kept_property_lines = 0
            if include_properties:
                output_lines.append(line)
            continue

        if line.startswith("functional group differences:"):
            section = "functional_groups"
            if include_functional_groups:
                output_lines.append(line)
            continue

        if section == "properties":
            if line == "" or line.startswith("Neighbor ") or line in {"positive neighbors:", "negative neighbors:"}:
                section = None
                output_lines.append(line)
                continue
            if include_properties and (property_lines is None or kept_property_lines < property_lines):
                output_lines.append(line)
            kept_property_lines += 1
            continue

        if section == "functional_groups":
            if line == "":
                section = None
                output_lines.append(line)
                continue
            if line.startswith("Neighbor ") or line in {"positive neighbors:", "negative neighbors:"}:
                section = None
                output_lines.append(line)
                continue
            if include_functional_groups:
                output_lines.append(line)
            continue

        output_lines.append(line)

    return "\n".join(output_lines)


def call_trim_tool(
    tool_call,
    *,
    task_name: str,
    neighbors_per_label: int,
    similar_tool_feature_view: str = "all",
    similar_tool_property_lines: int | None = None,
    chembl_include_activities: bool = True,
) -> str:
    if get_tool_call_name(tool_call) == "chembl_info":
        try:
            from chembl_info import chembl_info

            arguments = get_tool_call_arguments(tool_call)
            return chembl_info(
                str(arguments.get("smiles", "")),
                include_activities=chembl_include_activities,
            )
        except Exception as exc:
            return f"Error executing tool: {exc}"

    if _TOOL_RUNTIME is None:
        return "Error executing tool: TRIM tool runtime is not initialized."

    try:
        tool_result = _TOOL_RUNTIME.call_openai_function_call(
            tool_call,
            task=task_name,
            neighbors_per_label=neighbors_per_label,
        )
        if get_tool_call_name(tool_call) == "compare_similar_mols":
            tool_result = postprocess_compare_similar_mols_text(
                tool_result,
                feature_view=similar_tool_feature_view,
                property_lines=similar_tool_property_lines,
            )
        return tool_result
    except Exception as exc:
        return f"Error executing tool: {exc}"


def get_response_function_call_id(tool_call) -> str:
    if isinstance(tool_call, Mapping):
        return str(tool_call.get("call_id") or tool_call.get("id"))
    return str(getattr(tool_call, "call_id", None) or getattr(tool_call, "id"))


def run_openai_responses_turn(
    client,
    input_items,
    *,
    model_name,
    tools,
    task_name,
    neighbors_per_label,
    args,
):
    depth_limit = 40
    total_tool_calls = 0
    usage_cost = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
    }

    for _ in range(depth_limit):
        try:
            response = create_openai_response(
                client,
                model_name=model_name,
                input_items=input_items,
                tools=tools,
                args=args,
            )
        except Exception as exc:
            logger.error(f"Error in OpenAI Responses API call: {exc}")
            return None, total_tool_calls, usage_cost, normalize_messages_for_trace(input_items)

        call_usage_cost = extract_usage_cost(response, provider="openai", args=args)
        usage_cost["prompt_tokens"] += int(call_usage_cost["prompt_tokens"] or 0)
        usage_cost["completion_tokens"] += int(call_usage_cost["completion_tokens"] or 0)
        usage_cost["total_tokens"] += int(call_usage_cost["total_tokens"] or 0)
        usage_cost["reasoning_tokens"] += int(call_usage_cost["reasoning_tokens"] or 0)
        if call_usage_cost["cost_usd"] is not None:
            usage_cost["cost_usd"] += float(call_usage_cost["cost_usd"])

        function_outputs = []
        for output_item in response.output:
            if output_item.type != "function_call":
                continue
            total_tool_calls += 1
            tool_result = call_trim_tool(
                output_item,
                task_name=task_name,
                neighbors_per_label=neighbors_per_label,
                similar_tool_feature_view=args.similar_tool_feature_view,
                similar_tool_property_lines=args.similar_tool_property_lines,
                chembl_include_activities=args.chembl_include_activities,
            )
            function_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": get_response_function_call_id(output_item),
                    "output": tool_result,
                }
            )

        if not function_outputs:
            trace_items = list(input_items) + list(response.output)
            return response.output_text or "", total_tool_calls, usage_cost, normalize_messages_for_trace(trace_items)

        input_items = list(input_items) + list(response.output) + function_outputs

    return "", total_tool_calls, usage_cost, normalize_messages_for_trace(input_items)


def run_turn_base(
    client,
    messages,
    *,
    provider,
    model_name,
    tools,
    task_name,
    neighbors_per_label,
    args,
):
    if provider == "openai" and tools is not None:
        input_items = [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in messages
            if message.get("role") in {"user", "system", "developer"} and message.get("content") is not None
        ]
        return run_openai_responses_turn(
            client,
            input_items,
            model_name=model_name,
            tools=tools,
            task_name=task_name,
            neighbors_per_label=neighbors_per_label,
            args=args,
        )

    depth_limit = 40
    total_tool_calls = 0
    final_content = ""
    usage_cost = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
    }

    for _ in range(depth_limit):
        try:
            response = create_chat_completion(
                client,
                provider=provider,
                model_name=model_name,
                messages=messages,
                tools=tools,
                args=args,
            )
        except Exception as exc:
            logger.error(f"Error in chat completion: {exc}")
            return None, total_tool_calls, usage_cost, normalize_messages_for_trace(messages)

        call_usage_cost = extract_usage_cost(response, provider=provider, args=args)
        usage_cost["prompt_tokens"] += int(call_usage_cost["prompt_tokens"] or 0)
        usage_cost["completion_tokens"] += int(call_usage_cost["completion_tokens"] or 0)
        usage_cost["total_tokens"] += int(call_usage_cost["total_tokens"] or 0)
        usage_cost["reasoning_tokens"] += int(call_usage_cost["reasoning_tokens"] or 0)
        if call_usage_cost["cost_usd"] is not None:
            usage_cost["cost_usd"] += float(call_usage_cost["cost_usd"])

        message = response.choices[0].message
        messages.append(message)
        final_content = message.content or ""

        tool_calls = message.tool_calls or []
        if not tool_calls:
            return final_content, total_tool_calls, usage_cost, normalize_messages_for_trace(messages)

        total_tool_calls += len(tool_calls)
        for tool_call in tool_calls:
            tool_result = call_trim_tool(
                tool_call,
                task_name=task_name,
                neighbors_per_label=neighbors_per_label,
                similar_tool_feature_view=args.similar_tool_feature_view,
                similar_tool_property_lines=args.similar_tool_property_lines,
                chembl_include_activities=args.chembl_include_activities,
            )
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            }
            if provider != "deepseek":
                tool_message["name"] = tool_call.function.name
            messages.append(tool_message)

    return final_content, total_tool_calls, usage_cost, normalize_messages_for_trace(messages)


def render_user_text(task_name: str, smiles: str) -> str:
    from trim.reasoning.task_user_prompts import render_task_user_message

    return render_task_user_message(task=task_name, smiles=smiles)


def maybe_append_smiles_identity_guard(user_text: str, enabled: bool) -> str:
    if not enabled:
        return user_text
    return f"{user_text.rstrip()}\n\nAdditional instruction: {SMILES_IDENTITY_HEURISTIC_GUARD}"


def get_smiles(item: Mapping[str, Any]) -> str:
    for key in ("drug", "smiles", "SMILES"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError(f"Could not find SMILES field in record keys: {sorted(item.keys())}")


def worker_process_sample(payload):
    global _CLIENT

    if len(payload) == 5:
        index, sample_id, item, task_name, args_dict = payload
    else:
        index, item, task_name, args_dict = payload
        sample_id = 0
    args = argparse.Namespace(**args_dict)
    label = int(item["Y"])
    smiles = get_smiles(item)
    prompt_task = args.prompt_task or task_name
    user_text = render_user_text(prompt_task, smiles)
    user_text = maybe_append_smiles_identity_guard(user_text, args.disallow_smiles_identity_heuristics)
    messages = [{"role": "user", "content": user_text}]
    api_style = "responses" if args.provider == "openai" and args.tool_mode != "none" else "chat"
    tools = get_trim_tools_for_mode(args.tool_mode, api_style=api_style)

    last_tool_count = 0
    last_usage_cost = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
    }
    last_trace_messages = normalize_messages_for_trace(messages)
    last_response_text = ""

    def build_trajectory_record(attempt: int, prediction: int | None) -> dict[str, Any]:
        return {
            "task": task_name,
            "prompt_task": prompt_task,
            "index": index,
            "sample_id": sample_id,
            "attempt": attempt,
            "smiles": smiles,
            "label": label,
            "prediction": prediction,
            "tool_count": last_tool_count,
            "usage": last_usage_cost,
            "response_text": last_response_text,
            "reasoning_summaries": extract_reasoning_summaries(last_trace_messages),
            "messages": last_trace_messages,
        }

    for attempt in range(args.max_retry):
        current_messages = [dict(message) for message in messages]
        response_text, tool_count, usage_cost, trace_messages = run_turn_base(
            _CLIENT,
            current_messages,
            provider=args.provider,
            model_name=args.model,
            tools=tools,
            task_name=prompt_task,
            neighbors_per_label=args.neighbors_per_label,
            args=args,
        )
        last_tool_count = tool_count
        last_usage_cost = usage_cost
        last_trace_messages = trace_messages
        last_response_text = response_text or ""
        if not response_text:
            continue

        ans_txt, fmt_ok = extract_answer(response_text)
        prediction = parse_answer(ans_txt, fmt_ok, think_is_on=args.thinking, score_based=False)
        if prediction is not None:
            return index, prediction, tool_count, usage_cost, build_trajectory_record(attempt + 1, prediction)
        if getattr(args, "debug_parse_failures", False):
            excerpt = response_text.replace("\n", "\\n")[:1000]
            logger.warning(
                f"{task_name} idx={index} parse failed on attempt {attempt + 1}/{args.max_retry}; "
                f"tool_count={tool_count}; response excerpt: {excerpt}"
            )

    return index, None, last_tool_count, last_usage_cost, build_trajectory_record(args.max_retry, None)


def load_tasks_map(data_dir: Path) -> dict[str, list[str]]:
    mapping = {
        "Tox": [
            "Skin_Reaction.jsonl",
            "hERG.jsonl",
            "DILI.jsonl",
            "ClinTox.jsonl",
            "AMES.jsonl",
            "Carcinogens_Lagunin.jsonl",
        ],
        "ADME": [
            "PAMPA_NCATS.jsonl",
            "HIA_Hou.jsonl",
            "BBB_Martins.jsonl",
            "Pgp_Broccatelli.jsonl",
            "Bioavailability_Ma.jsonl",
            "CYP2C9_Substrate_CarbonMangels.jsonl",
            "CYP2D6_Substrate_CarbonMangels.jsonl",
            "CYP3A4_Substrate_CarbonMangels.jsonl",
        ],
        "HTS": [
            "SARSCoV2_3CLPro_Diamond.jsonl",
            "SARSCoV2_Vitro_Touret.jsonl",
        ],
    }
    available_files = {path.name for path in data_dir.glob("*.jsonl")}
    return {
        group: [filename for filename in filenames if filename in available_files]
        for group, filenames in mapping.items()
        if any(filename in available_files for filename in filenames)
    }


def configure_file_logging(args) -> None:
    if not args.log_file:
        return
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / args.run_name
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging to file: {log_path}")


def configure_reasoning_trajectory_dir(args) -> Path | None:
    if not args.save_reasoning_trajectories:
        return None
    trajectory_dir = project_root / "reasoning-trajectory" / args.run_name
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving reasoning trajectories to: {trajectory_dir}")
    return trajectory_dir


def write_task_trajectories(output_dir: Path, task_name: str, records: list[dict[str, Any]]) -> None:
    output_path = output_dir / f"{task_name}.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: (int(item["index"]), int(item["sample_id"]))):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records)} reasoning trajectories to {output_path}")


def main():
    args = get_args()
    try:
        args = resolve_api_settings(args)
    except ValueError as exc:
        logger.error(str(exc))
        return

    args.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_name = args.log_file_name.format(t_stamp=args.run_timestamp)
    configure_file_logging(args)
    reasoning_trajectory_dir = configure_reasoning_trajectory_dir(args)

    if not args.data_dir.exists():
        logger.error(f"Data directory does not exist: {args.data_dir}")
        return

    if not args.model:
        try:
            temp_client = OpenAI(api_key=args.api_key, base_url=args.api_base)
            models = temp_client.models.list()
            args.model = models.data[0].id
            logger.info(f"Detected model: {args.model}")
        except Exception as exc:
            logger.error(f"Could not connect to API to detect model: {exc}")
            return

    task_map = load_tasks_map(args.data_dir)
    if args.tasks:
        task_map = {
            "custom": [
                f"{task}.jsonl" if not str(task).endswith(".jsonl") else str(task)
                for task in args.tasks
                if (args.data_dir / (f"{task}.jsonl" if not str(task).endswith(".jsonl") else str(task))).exists()
            ]
        }
        groups_to_run = ["custom"]
    else:
        groups_to_run = list(task_map) if "all" in args.task_groups else args.task_groups

    logger.info(f"Provider: {args.provider}")
    logger.info(f"API base: {args.api_base}")
    logger.info(f"Model: {args.model}")
    if args.provider == "openai":
        logger.info(f"OpenAI reasoning effort: {args.openai_reasoning_effort}")
        logger.info(f"OpenAI reasoning summary: {args.openai_reasoning_summary}")
    elif args.provider == "deepseek":
        logger.info(f"DeepSeek thinking: {args.thinking}")
        if args.thinking:
            logger.info(f"DeepSeek reasoning effort: {args.deepseek_reasoning_effort}")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Tool mode: {args.tool_mode}")
    logger.info(f"Disallow SMILES identity heuristics: {args.disallow_smiles_identity_heuristics}")
    logger.info(f"Multiprocessing start method: {multiprocessing.get_start_method()}")
    if args.tool_mode in CHEMBL_TOOL_MODES:
        logger.info(f"chembl_info include activities: {args.chembl_include_activities}")
    if "compare_similar_mols" in TRIM_TOOL_NAMES_BY_MODE[args.tool_mode]:
        logger.info(f"Similar tool feature view: {args.similar_tool_feature_view}")
        logger.info(f"Similar tool property lines: {args.similar_tool_property_lines or 'all'}")
    logger.info(f"Running groups: {groups_to_run}")

    all_results: dict[str, dict[str, float]] = {}
    total_usage_cost = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
    }
    args_dict = vars(args).copy()

    for group in groups_to_run:
        if group not in task_map:
            logger.warning(f"No files found for group {group} or group not defined.")
            continue

        all_results[group] = {}
        for filename in task_map[group]:
            file_path = args.data_dir / filename
            task_name = file_path.stem
            logger.info(f"Processing Task: {task_name}")

            raw_data = []
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        raw_data.append(json.loads(line))

            if args.limit_samples is not None:
                raw_data = raw_data[: args.limit_samples]

            if not raw_data:
                logger.warning(f"No data in {filename}")
                continue

            worker_tasks = []
            for idx, item in enumerate(raw_data):
                for sample_id in range(args.n_samples):
                    worker_tasks.append((idx, sample_id, item, task_name, args_dict))

            results = []
            with multiprocessing.Pool(
                processes=min(args.num_processes, len(worker_tasks)),
                initializer=init_worker,
                initargs=(args.api_base, args.api_key, args.provider, args.tool_mode),
            ) as pool:
                for result in tqdm(
                    pool.imap_unordered(worker_process_sample, worker_tasks),
                    total=len(worker_tasks),
                    desc=task_name,
                ):
                    results.append(result)

            item_results: dict[int, list[int]] = {}
            item_tool_counts: dict[int, list[int]] = {}
            if reasoning_trajectory_dir is not None:
                task_trajectory_records = [trajectory for _, _, _, _, trajectory in results]
                write_task_trajectories(reasoning_trajectory_dir, task_name, task_trajectory_records)

            none_pred_count = sum(1 for _, pred, _, _, _ in results if pred is None)
            logger.info(
                f"{task_name} Failure Rate (pred is None): "
                f"{none_pred_count}/{len(results)} ({none_pred_count / len(results) * 100:.2f}%)"
            )

            task_usage_cost = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "cost_usd": 0.0,
            }

            for idx, pred, tool_count, usage_cost, _ in results:
                for key in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"):
                    task_usage_cost[key] += int(usage_cost.get(key, 0) or 0)
                task_usage_cost["cost_usd"] += float(usage_cost.get("cost_usd", 0.0) or 0.0)

                item_results.setdefault(idx, [])
                item_tool_counts.setdefault(idx, [])
                if pred is not None:
                    item_results[idx].append(pred)
                    item_tool_counts[idx].append(tool_count)

            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"):
                total_usage_cost[key] += task_usage_cost[key]
            total_usage_cost["cost_usd"] += task_usage_cost["cost_usd"]

            y_true = []
            y_pred = []
            failed_parses_count = 0
            for idx, item in enumerate(raw_data):
                label = int(item["Y"])
                y_true.append(label)
                preds = item_results.get(idx, [])
                if not preds:
                    failed_parses_count += 1
                    y_pred.append(1 - label)
                    continue
                y_pred.append(1 if preds.count(1) >= preds.count(0) else 0)

            logger.info("\n" + "=" * 80)
            logger.info(f"{task_name} EVALUATION RESULTS (Macro F1)")
            logger.info("=" * 80)
            logger.info(f"Failed parses (all samples failed for a q): {failed_parses_count}/{len(raw_data)}")

            if len(set(y_true)) > 1:
                score = f1_score(y_true, y_pred, average="macro")
                logger.info(f"Classification Report:\n{classification_report(y_true, y_pred, digits=4, labels=[0, 1])}")
                logger.info(f"Macro F1 Score: {score:.4f}")
                all_results[group][task_name] = score
            else:
                logger.info(f"{task_name} F1: Cannot calculate (one class only in truth)")
                all_results[group][task_name] = 0.0

            if args.tool_mode != "none":
                all_tool_counts = [count for counts in item_tool_counts.values() for count in counts]
                questions_with_tool_usage = sum(1 for counts in item_tool_counts.values() if any(count > 0 for count in counts))
                total_questions = sum(1 for counts in item_tool_counts.values() if counts)
                avg_tool_calls = float(np.mean(all_tool_counts)) if all_tool_counts else 0.0
                usage_rate = (questions_with_tool_usage / total_questions * 100) if total_questions else 0.0
                logger.info(f"{task_name} Avg Tools/Sample: {avg_tool_calls:.2f}")
                logger.info(f"{task_name} Questions w/ Tools: {questions_with_tool_usage}/{total_questions} ({usage_rate:.1f}%)")

            logger.info(
                f"{task_name} API usage: input={task_usage_cost['prompt_tokens']} "
                f"output={task_usage_cost['completion_tokens']} "
                f"reasoning={task_usage_cost['reasoning_tokens']} "
                f"total={task_usage_cost['total_tokens']} "
                f"estimated/reported cost=${task_usage_cost['cost_usd']:.6f}"
            )

            logger.info("=" * 80 + "\n")

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY RESULTS (Macro F1)")
    logger.info("=" * 80)
    for group in groups_to_run:
        group_results = all_results.get(group, {})
        if not group_results:
            continue
        logger.info(f"\n{group} Tasks:")
        group_scores = []
        for task_name, score in group_results.items():
            logger.info(f"  {task_name}: {score:.4f}")
            group_scores.append(score)
        logger.info(f"  Average: {np.mean(group_scores):.4f}")
    logger.info(
        f"\nTotal API usage: input={total_usage_cost['prompt_tokens']} "
        f"output={total_usage_cost['completion_tokens']} "
        f"reasoning={total_usage_cost['reasoning_tokens']} "
        f"total={total_usage_cost['total_tokens']} "
        f"estimated/reported cost=${total_usage_cost['cost_usd']:.6f}"
    )
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    start_method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
    multiprocessing.set_start_method(start_method, force=True)
    main()
