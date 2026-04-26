import os
import time
import json
import requests
from statistics import mean, stdev

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8001/v1")
API_KEY = os.environ.get("API_KEY", "EMPTY")
MODEL = os.environ.get("MODEL", "gemma-4-26B-A4B-it")

URL = f"{BASE_URL.rstrip('/')}/chat/completions"

PROMPT = """Write a concise technical explanation of why transformer inference becomes memory-bandwidth limited during autoregressive decoding. 
Use about 500 words."""

N_RUNS = 5
MAX_TOKENS = 512


def run_once(run_id: int):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": PROMPT}
        ],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    t0 = time.perf_counter()
    ttft = None
    text_chunks = []
    usage = None

    with requests.post(URL, headers=headers, json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()

        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue

            if line.startswith("data: "):
                line = line[len("data: "):]

            if line.strip() == "[DONE]":
                break

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            # vLLM/OpenAI streaming final chunk may contain usage
            if data.get("usage") is not None:
                usage = data["usage"]

            choices = data.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            content = delta.get("content")

            if content:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text_chunks.append(content)

    t1 = time.perf_counter()
    elapsed = t1 - t0

    output_text = "".join(text_chunks)

    if usage is not None:
        prompt_tokens = usage.get("prompt_tokens", None)
        completion_tokens = usage.get("completion_tokens", None)
        total_tokens = usage.get("total_tokens", None)
    else:
        # fallback: approximate by character length; less accurate
        prompt_tokens = None
        completion_tokens = max(1, len(output_text) // 4)
        total_tokens = None

    decode_time = elapsed - (ttft or 0.0)
    completion_tps = completion_tokens / decode_time if decode_time > 0 else float("nan")
    e2e_completion_tps = completion_tokens / elapsed if elapsed > 0 else float("nan")
    total_tps = total_tokens / elapsed if total_tokens and elapsed > 0 else None

    result = {
        "run": run_id,
        "ttft_s": ttft,
        "elapsed_s": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "decode_completion_tok_s": completion_tps,
        "e2e_completion_tok_s": e2e_completion_tps,
        "total_tok_s": total_tps,
    }

    return result


def fmt(x):
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def main():
    print(f"BASE_URL = {BASE_URL}")
    print(f"MODEL    = {MODEL}")
    print(f"N_RUNS   = {N_RUNS}")
    print(f"MAX_TOKENS = {MAX_TOKENS}")
    print()

    # Warmup
    print("Warmup...")
    _ = run_once(0)
    print("Warmup done.\n")

    results = []
    for i in range(1, N_RUNS + 1):
        res = run_once(i)
        results.append(res)
        print(
            f"run={res['run']} | "
            f"TTFT={fmt(res['ttft_s'])}s | "
            f"elapsed={fmt(res['elapsed_s'])}s | "
            f"prompt_tok={fmt(res['prompt_tokens'])} | "
            f"completion_tok={fmt(res['completion_tokens'])} | "
            f"decode_tok/s={fmt(res['decode_completion_tok_s'])} | "
            f"e2e_completion_tok/s={fmt(res['e2e_completion_tok_s'])} | "
            f"total_tok/s={fmt(res['total_tok_s'])}"
        )

    def avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return mean(vals) if vals else None

    def sd(key):
        vals = [r[key] for r in results if r[key] is not None]
        return stdev(vals) if len(vals) > 1 else 0.0

    print("\n=== Summary ===")
    for key in [
        "ttft_s",
        "elapsed_s",
        "prompt_tokens",
        "completion_tokens",
        "decode_completion_tok_s",
        "e2e_completion_tok_s",
        "total_tok_s",
    ]:
        print(f"{key}: mean={fmt(avg(key))}, std={fmt(sd(key))}")


if __name__ == "__main__":
    main()