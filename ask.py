import json
import math
import urllib.request
from pathlib import Path

BASE = Path.home() / "AI-System"
INDEX = BASE / "index/test_embedding.json"

def embed(text):
    data = json.dumps({
        "model": "text-embedding-nomic-embed-text-v1.5",
        "input": text
    }).encode("utf-8")

    req = urllib.request.Request(
        "http://127.0.0.1:1234/v1/embeddings",
        data=data,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req) as r:
        return json.load(r)["data"][0]["embedding"]

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot / (na * nb)

def ask_gpt(question, memory):
    prompt = f"""你是我的本地 AI 助手。

下面是从我的本地长期记忆中检索到的信息：

--- 本地记忆 ---
{memory}
--- 本地记忆结束 ---

请只根据相关记忆回答用户问题。
如果记忆中没有答案，就明确说不知道，不要编造。

用户问题：
{question}
"""

    data = json.dumps({
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }).encode("utf-8")

    req = urllib.request.Request(
        "http://127.0.0.1:1234/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req) as r:
        result = json.load(r)

    return result["choices"][0]["message"]["content"]

question = " ".join(__import__("sys").argv[1:]).strip()

if not question:
    print('用法：python3 ~/AI-System/ask.py "你的问题"')
    raise SystemExit

record = json.loads(INDEX.read_text(encoding="utf-8"))

query_vector = embed(question)
score = cosine(query_vector, record["embedding"])

print(f"🔎 检索相似度: {score:.4f}")
print(f"📄 找到记忆: {record['source']}")
print()

answer = ask_gpt(question, record["text"])

print("🤖 GPT-OSS:")
print(answer)
