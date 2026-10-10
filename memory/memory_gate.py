import json
import urllib.request

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"


SYSTEM_PROMPT = """
你是一个本地 AI 长期记忆判断器（Memory Gate）。

你的唯一任务：
判断用户当前说的话，是否值得保存为长期记忆。

长期记忆应该保存：
- 用户长期稳定的偏好
- 用户明确表达的长期要求
- 用户长期使用的工具、设备、系统
- 用户正在进行的重要长期项目
- 用户做出的重要长期决定
- 对未来多次对话有帮助的信息

不要保存：
- 普通闲聊
- 一次性的临时状态
- 当前正在发生但很快会失效的信息
- 单纯的问题
- 普通事实说明
- 没有长期价值的操作过程
- 重复且没有新信息的内容

【长期事实识别规则】
1. 用户明确陈述的长期项目配置、使用的工具或模型、开发原则、隐私要求，即使没有说“请记住”，也应当保存。
2. 包含“以后”“始终”“长期”“开发过程中”等表达的稳定原则，应优先判断为长期记忆。
3. 不要把陈述误判为普通问题，也不要要求用户使用特定的记忆触发词。
4. 只有真正重复、临时或没有未来价值的信息才不保存。

特别注意：
“用户希望……”
“用户以后……”
“用户长期……”
“用户决定……”
通常具有较高的长期记忆价值。

你必须只返回 JSON，不要输出任何解释。

JSON 格式必须严格为：

{
  "should_remember": true,
  "type": "preferences",
  "importance": 0.0,
  "confidence": 0.0,
  "memory": "适合长期保存的一句话",
  "reason": "简短原因"
}

type 只能是：
core
preferences
projects
experiences
decisions

importance 范围：0.0 到 1.0
confidence 范围：0.0 到 1.0

如果不值得记忆：

{
  "should_remember": false,
  "type": null,
  "importance": 0.0,
  "confidence": 0.0,
  "memory": null,
  "reason": "简短原因"
}
"""


def ask_gate(user_text):
    payload = {
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_text
            }
        ]
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    request = urllib.request.Request(
        LM_STUDIO_URL,
        data=data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(
            response.read().decode("utf-8")
        )

    content = result["choices"][0]["message"]["content"]

    return content


def main():
    import sys

    if len(sys.argv) < 2:
        print("用法：")
        print('python3 memory_gate.py "用户说的话"')
        return

    user_text = " ".join(sys.argv[1:])

    print("\nMemory Gate")
    print("=" * 60)
    print("用户内容：")
    print(user_text)
    print("=" * 60)

    raw = ask_gate(user_text)

    print("GPT-OSS 原始判断：")
    print(raw)

    print("\n解析结果：")

    try:
        result = json.loads(raw)

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2
            )
        )

    except json.JSONDecodeError:
        print("WARNING: GPT-OSS 没有返回有效 JSON。")


if __name__ == "__main__":
    main()
