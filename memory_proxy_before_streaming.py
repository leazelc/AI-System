import json
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys

HOST = "127.0.0.1"
PORT = 8766

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-20b"

BASE = Path.home() / "AI-System"
MEMORY_ROOT = BASE / "memory"

sys.path.insert(0, str(MEMORY_ROOT))

from memory_search import search
from memory_writer_v2 import execute as write_memory


def call_lm_studio(messages, model):
    data = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.2
    }).encode("utf-8")

    req = urllib.request.Request(
        LM_STUDIO_URL,
        data=data,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(req) as response:
        return json.load(response)


def build_memory_prompt(question, memories):
    if memories:
        memory_text = "\n\n".join(
            f"""--- 长期记忆 {i} ---
类型：{item["metadata"].get("type", "unknown")}

{item["content"]}"""
            for i, item in enumerate(memories, 1)
        )
    else:
        memory_text = "没有检索到相关长期记忆。"

    return f"""你是运行在用户本地电脑上的 AI 助手。

你拥有一个本地长期记忆系统。

以下是根据当前问题检索出的相关长期记忆：

========== 长期记忆 ==========
{memory_text}
========== 长期记忆结束 ==========

请正常回答用户的问题。

规则：
1. 相关记忆可以自然使用。
2. 不相关记忆不要强行引用。
3. 不要编造不存在的记忆。
4. 不要向用户暴露 Memory Core、SQLite、embedding 等内部实现。

用户问题：
{question}
"""


class ProxyHandler(BaseHTTPRequestHandler):

    def send_json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {
                "status": "ok",
                "service": "memory-proxy",
                "model": DEFAULT_MODEL
            })
            return

        if self.path == "/v1/models":
            self.send_json(200, {
                "object": "list",
                "data": [{
                    "id": DEFAULT_MODEL,
                    "object": "model",
                    "owned_by": "local-memory-proxy"
                }]
            })
            return

        self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            request = json.loads(raw.decode("utf-8"))

            messages = request.get("messages", [])
            model = request.get("model", DEFAULT_MODEL)

            if not messages:
                self.send_json(400, {
                    "error": "messages is required"
                })
                return

            user_messages = [
                m for m in messages
                if m.get("role") == "user"
            ]

            if not user_messages:
                self.send_json(400, {
                    "error": "No user message found"
                })
                return

            question = user_messages[-1].get("content", "")

            if not isinstance(question, str):
                question = str(question)

            print(f"[Memory Proxy] User: {question}")

            memories = search(question, limit=5)

            print(
                f"[Memory Proxy] Memories found: {len(memories)}"
            )

            memory_prompt = build_memory_prompt(
                question,
                memories
            )

            upstream_messages = [
                {
                    "role": "user",
                    "content": memory_prompt
                }
            ]

            result = call_lm_studio(
                upstream_messages,
                DEFAULT_MODEL
            )

            answer = result["choices"][0]["message"]["content"]

            try:
                write_memory(question)
            except Exception as e:
                print(
                    f"[Memory Proxy] Memory Writer error: {e}"
                )

            result["model"] = model

            self.send_json(200, result)

        except Exception as e:
            print(f"[Memory Proxy] ERROR: {e}")

            self.send_json(500, {
                "error": str(e)
            })

    def log_message(self, format, *args):
        print("[Proxy]", format % args)


def main():
    server = HTTPServer((HOST, PORT), ProxyHandler)

    print("================================")
    print("Memory Proxy")
    print("================================")
    print(f"Listening: http://{HOST}:{PORT}")
    print(f"Chat API:  http://{HOST}:{PORT}/v1/chat/completions")
    print(f"Models:    http://{HOST}:{PORT}/v1/models")
    print("================================")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Memory Proxy...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
