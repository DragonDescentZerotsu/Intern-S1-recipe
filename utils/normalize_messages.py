import json

def dump_obj(x):
    if hasattr(x, "model_dump"):
        return x.model_dump(exclude_none=True, mode="json")
    if hasattr(x, "dict"):
        return x.dict()
    return x

def normalize_DeepSeek_V32_message(msg_obj):
    m = dump_obj(msg_obj)

    role = m.get("role")

    # ---------- 通用：content 归一化 ----------
    # 1) None -> ""
    if m.get("content") is None:
        m["content"] = ""

    # 2) 如果 content 是 dict/list（比如某些 tool 返回），尽量转成字符串
    if role == "tool":
        # tool 的 content 最稳就是字符串；非字符串就 JSON 化
        if not isinstance(m["content"], str):
            m["content"] = json.dumps(m["content"], ensure_ascii=False)

    # ---------- 按 role 分支 ----------
    if role == "assistant":
        # 统一 reasoning 字段名：优先 reasoning_content，其次 reasoning
        if not isinstance(m.get("reasoning_content"), str):
            # dump 结果里可能直接有 reasoning
            if isinstance(m.get("reasoning"), str):
                m["reasoning_content"] = m["reasoning"]
            else:
                # 兜底：从 model_extra 里找（有些 SDK 会放这里）
                extra = getattr(msg_obj, "model_extra", None) or {}
                m["reasoning_content"] = extra.get("reasoning_content") or extra.get("reasoning")

        # tool_calls：确保是 list[dict]（你模板能吃 dict）
        if m.get("tool_calls"):
            # 如果 dump_obj 已经把 tool_calls dump 成 dict list，下面这段可省略
            normalized = []
            for tc in m["tool_calls"]:
                tc_d = dump_obj(tc)
                # 兼容一些实现：function 可能叫 "function" 或 "tool" 等，这里只保证你模板需要的字段存在
                if "type" not in tc_d:
                    tc_d["type"] = "function"
                normalized.append(tc_d)
            m["tool_calls"] = normalized

    return m

def normalize_DeepSeek_V32_messages(messages):
    return [normalize_DeepSeek_V32_message(msg) for msg in messages]

def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
