import json
import urllib.request
from pathlib import Path
import sys

# ============================================================
# Memory Gateway
# 第一阶段：只接入 GPT-OSS 20B
# 未来可通过 Adapter 接入其他本地 AI
# ============================================================

BASE = Path.home() / "AI-System"
MEMORY_ROOT = BASE / "memory"

# ------------------------------------------------------------
# AI Adapter 配置
# ------------------------------------------------------------

ADAPTERS = {
    "gpt-oss": {
        "enabled": True,
        "url": "http://127.0.0.1:1234/v1/chat/completions",
        "model": "openai/gpt-oss-20b",
    },

    # 未来可以在这里增加：
    #
    # "gemma": {
    #     "enabled": False,
    #     "url": "...",
    #     "model": "google/gemma-4-12b",
    # },
    #
    # "codex": {
    #     "enabled": False,
    # },
}


# ------------------------------------------------------------
# Memory Core
# ------------------------------------------------------------

sys.path.insert(0, str(MEMORY_ROOT))

from memory_search import search
from memory_writer_v2 import execute as write_memory


# ------------------------------------------------------------
# GPT-OSS Adapter
# ------------------------------------------------------------

def gpt_oss_adapter(question, memories):
    """
    GPT-OSS 的统一 Adapter。
    以后其他模型只需要增加自己的 Adapter，
    不需要修改 Memory Core。
    """

    config = ADAPTERS["gpt-oss"]

    if not config["enabled"]:
        raise RuntimeError("GPT-OSS adapter is disabled")

    if memories:
        memory_text = "\n\n".join(
            f"""--- 长期记忆 {i} ---
类型：{item["metadata"].get("type", "unknown")}

{item["content"]}"""
            for i, item in enumerate(memories, 1)
        )
    else:
        memory_text = "没有检索到相关长期记忆。"

    prompt = f"""你是运行在用户本地电脑上的 AI 助手。

你拥有一个本地长期记忆系统。

下面是根据当前问题检索出来的相关长期记忆：

========== 长期记忆 ==========
{memory_text}
========== 长期记忆结束 ==========

请自然地回答用户的问题。

规则：
1. 长期记忆只作为用户背景信息使用。
2. 如果长期记忆与当前问题无关，不要强行引用。
3. 不要编造不存在的记忆。
4. 如果记忆与问题相关，可以自然利用。
5. 不要向用户暴露 Memory Core 的内部实现。
6. 你运行在用户本地电脑上。

用户问题：
{question}
"""

    data = json.dumps({
        "model": config["model"],
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2
    }).encode("utf-8")

    req = urllib.request.Request(
        config["url"],
        data=data,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req) as response:
        result = json.load(response)

    return result["choices"][0]["message"]["content"]


# ------------------------------------------------------------
# Gateway
# ------------------------------------------------------------

def ask(question):
    """
    Memory Gateway 主入口：

    用户输入
        ↓
    Memory Search
        ↓
    GPT-OSS Adapter
        ↓
    返回回答
        ↓
    Memory Writer
    """

    memories = search(question, limit=5)

    answer = gpt_oss_adapter(question, memories)

    try:
        write_memory(question)
    except Exception as e:
        print(
            f"⚠️ 长期记忆系统出现错误，但不影响本次回答：{e}"
        )

    return answer


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        print('用法：python3 ~/AI-System/memory_gateway.py "你的问题"')
        raise SystemExit(1)

    print("🔎 Memory Gateway")
    print("📚 正在检索长期记忆...")

    memories = search(question, limit=5)
    print(f"📚 找到 {len(memories)} 条相关记忆")

    print("🤖 正在调用 GPT-OSS...\n")

    answer = gpt_oss_adapter(question, memories)

    print("GPT-OSS:")
    print(answer)

    print("\n🧠 正在判断是否需要保存长期记忆...")

    try:
        write_memory(question)
    except Exception as e:
        print(
            f"⚠️ 长期记忆系统出现错误，但不影响本次回答：{e}"
        )


if __name__ == "__main__":
    main()
