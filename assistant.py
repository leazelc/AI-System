import json
import sys
import urllib.request
from pathlib import Path

# AI-System 根目录
BASE = Path.home() / "AI-System"

# 本地 LM Studio API
LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"

# 导入长期记忆检索器
sys.path.insert(0, str(BASE / "memory"))
from memory_search import search
from memory_writer_v2 import execute as write_memory




def ask_gpt(question, memories):
    """
    将用户问题 + 相关长期记忆发送给 GPT-OSS。
    """

    if memories:
        memory_text = "\n\n".join(
            f"""--- 记忆 {i} ---
ID: {item["metadata"].get("id", "unknown")}
类型: {item["metadata"].get("type", "unknown")}
相关度: {item["score"]:.4f}

{item["content"]}"""
            for i, item in enumerate(memories, 1)
        )
    else:
        memory_text = "没有检索到相关长期记忆。"

    prompt = f"""你是我的本地 AI 助手。

你拥有一个本地长期记忆系统。

下面是根据当前问题，从我的本地长期记忆中检索出来的相关内容：

========== 长期记忆 ==========
{memory_text}
========== 长期记忆结束 ==========

请根据用户的问题回答。

重要规则：
1. 长期记忆只作为用户背景信息使用。
2. 如果长期记忆与当前问题无关，不要强行引用。
3. 不要编造长期记忆中不存在的信息。
4. 如果记忆与当前问题相关，可以自然地利用它。
5. 不要向用户暴露内部的 memory ID、similarity、SQLite、embedding 等实现细节。
6. 你是运行在用户本地电脑上的 AI 助手。

用户问题：
{question}
"""

    data = json.dumps({
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2
    }).encode("utf-8")

    req = urllib.request.Request(
        LM_STUDIO_URL,
        data=data,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req) as response:
        result = json.load(response)

    return result["choices"][0]["message"]["content"]


def main():
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        print('用法：python3 ~/AI-System/assistant.py "你的问题"')
        raise SystemExit(1)

    print("🔎 正在检索长期记忆...")

    memories = search(question, limit=5)

    print(f"📚 找到 {len(memories)} 条相关记忆")

    print("🤖 正在调用 GPT-OSS...\n")
    answer = ask_gpt(question, memories)

    print("GPT-OSS:")
    print(answer)

    # Automatic long-term memory path.
    # Memory failures must never break the normal answer path.
    print("\n🧠 正在判断是否需要保存长期记忆...")
    try:
        write_memory(question)
    except Exception as e:
        print(f"⚠️ 长期记忆系统出现错误，但不影响本次回答：{e}")


if __name__ == "__main__":
    main()
