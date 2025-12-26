import json

def save_jsonl(records: list[dict], path: str, ensure_ascii: bool = False):
    """
    records: list[dict]
    path: 输出文件路径，如 "data.jsonl"
    ensure_ascii=False: 保留中文不转义；True 则会变成 \\uXXXX
    """
    with open(path, "w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=ensure_ascii) + "\n")