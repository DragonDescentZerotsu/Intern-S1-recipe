from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
    timeout=180.0,     # 先拉大
    max_retries=0,     # 避免重试掩盖问题 :contentReference[oaicite:2]{index=2}
)

resp = client.chat.completions.create(
    model="local-model",
    messages=[{"role":"user","content":"ping"}],
    max_tokens=8,
    temperature=0,
)
print(resp.choices[0].message.content)
