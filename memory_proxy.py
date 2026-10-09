from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import urllib.request
import urllib.error
import sys

BASE = Path.home() / "AI-System"
MEMORY_ROOT = BASE / "memory"

sys.path.insert(0, str(MEMORY_ROOT))

from memory_search import search
from memory_writer_v2 import execute as write_memory


HOST = "127.0.0.1"
PORT = 8766

UPSTREAM_URL = "http://127.0.0.1:1234/v1/chat/completions"
UPSTREAM_MODEL = "openai/gpt-oss-20b"


def extract_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if not isinstance(item, dict):
                continue

            if item.get("type") == "text":
                text = item.get("text", "")
                if text:
                    parts.append(text)

        return "\n".join(parts)

    return str(content or "")


def get_latest_user_message(messages):
    for message in reversed(messages):
        if message.get("role") == "user":
            return extract_text(message.get("content", ""))

    return ""


def build_memory_context(question):
    try:
        memories = search(question, limit=5)
    except Exception as e:
        print(f"[Memory Proxy] Memory search error: {e}")
        memories = []

    if not memories:
        memory_text = "没有检索到相关长期记忆。"
    else:
        blocks = []

        for i, item in enumerate(memories, 1):
            metadata = item.get("metadata", {})
            content = item.get("content", "")

            blocks.append(
                f"""--- 记忆 {i} ---
类型: {metadata.get("type", "unknown")}
相关度: {item.get("score", 0):.4f}

{content}"""
            )

        memory_text = "\n\n".join(blocks)

    print(f"[Memory Proxy] Memories found: {len(memories)}")

    return f"""你是我的本地 AI 助手。

下面的信息来自我的本地长期记忆系统。
这些内容是已经保存并确认过的用户长期信息，不是需要你猜测或重新询问的信息。

========== 本地长期记忆 ==========
{memory_text}
========== 本地长期记忆结束 ==========

使用长期记忆时必须遵守：

1. 如果用户的问题直接询问某条长期记忆中的事实，直接使用该记忆回答。
2. 不要因为记忆内容已经存在，就反过来询问用户同样的信息。
3. 长期记忆中的具体数字、预算、设备、偏好、限制和长期决定，应视为用户已经提供并确认过的信息。
4. 只有当当前问题与某条记忆明显相关时才使用它。
5. 不相关的记忆不要影响回答。
6. 不要把“电脑预算上限为2万元人民币”理解成让用户重新提供预算；如果用户问电脑预算是多少，就直接回答2万元人民币。
7. 不要编造长期记忆中不存在的信息。
8. 不要向用户暴露 memory ID、SQLite、embedding、similarity 或内部实现。
9. 你运行在用户本地电脑上。

现在请直接回答用户的问题。

用户当前问题：
{question}
"""


def inject_memory(messages, memory_context):
    new_messages = [dict(message) for message in messages]

    if (
        new_messages
        and new_messages[0].get("role") == "system"
    ):
        original = extract_text(new_messages[0].get("content", ""))

        new_messages[0] = dict(new_messages[0])
        new_messages[0]["content"] = (
            original
            + "\n\n"
            + memory_context
        )
    else:
        new_messages.insert(
            0,
            {
                "role": "system",
                "content": memory_context,
            },
        )

    return new_messages


def call_writer(question):
    try:
        print("[Memory Proxy] Running Memory Writer...")
        write_memory(question)
    except Exception as e:
        print(f"[Memory Proxy] Writer error: {e}")


class MemoryProxyHandler(BaseHTTPRequestHandler):

    server_version = "MemoryProxy/1.0"

    def log_message(self, format, *args):
        print(f"[Proxy] {self.address_string()} - {format % args}")

    def send_json(self, status, data):
        body = json.dumps(
            data,
            ensure_ascii=False
        ).encode("utf-8")

        self.send_response(status)
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8"
        )
        self.send_header(
            "Content-Length",
            str(len(body))
        )
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send_json(
                200,
                {
                    "status": "ok",
                    "service": "memory-proxy",
                    "model": UPSTREAM_MODEL,
                },
            )
            return

        if self.path == "/v1/models":
            self.send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": UPSTREAM_MODEL,
                            "object": "model",
                            "owned_by": "local-memory-proxy",
                        }
                    ],
                },
            )
            return

        self.send_json(
            404,
            {"error": "Not found"},
        )

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_json(
                404,
                {"error": "Not found"},
            )
            return

        try:
            content_length = int(
                self.headers.get("Content-Length", "0")
            )

            raw_body = self.rfile.read(content_length)
            body = json.loads(raw_body.decode("utf-8"))

            messages = body.get("messages", [])

            if not messages:
                self.send_json(
                    400,
                    {"error": "messages is required"},
                )
                return

            question = get_latest_user_message(messages)

            if not question:
                self.send_json(
                    400,
                    {"error": "No user message found"},
                )
                return

            print(
                f'[Memory Proxy] User: {question}'
            )

            memory_context = build_memory_context(
                question
            )

            upstream_body = dict(body)

            requested_model = body.get(
                "model",
                UPSTREAM_MODEL
            )

            upstream_body["model"] = UPSTREAM_MODEL

            upstream_body["messages"] = inject_memory(
                messages,
                memory_context
            )

            stream = bool(
                upstream_body.get("stream", False)
            )

            request_data = json.dumps(
                upstream_body,
                ensure_ascii=False
            ).encode("utf-8")

            request = urllib.request.Request(
                UPSTREAM_URL,
                data=request_data,
                headers={
                    "Content-Type":
                        "application/json",
                    "Accept":
                        "text/event-stream"
                        if stream
                        else "application/json",
                },
                method="POST",
            )

            try:
                upstream = urllib.request.urlopen(
                    request,
                    timeout=600
                )
            except urllib.error.HTTPError as e:
                error_body = e.read().decode(
                    "utf-8",
                    errors="replace"
                )

                self.send_json(
                    e.code,
                    {
                        "error": error_body
                    },
                )
                return

            if stream:
                self.handle_stream(
                    upstream,
                    requested_model,
                    question
                )
            else:
                self.handle_non_stream(
                    upstream,
                    requested_model,
                    question
                )

        except Exception as e:
            print(
                f"[Memory Proxy] Request error: {e}"
            )

            try:
                self.send_json(
                    500,
                    {
                        "error": str(e)
                    },
                )
            except Exception:
                pass

    def handle_non_stream(
        self,
        upstream,
        requested_model,
        question
    ):
        with upstream:
            result = json.load(upstream)

        result["model"] = requested_model

        self.send_json(
            200,
            result
        )

        call_writer(question)

    def handle_stream(
        self,
        upstream,
        requested_model,
        question
    ):
        self.send_response(200)

        self.send_header(
            "Content-Type",
            "text/event-stream; charset=utf-8"
        )
        self.send_header(
            "Cache-Control",
            "no-cache"
        )
        self.send_header(
            "Connection",
            "close"
        )
        self.send_header(
            "X-Accel-Buffering",
            "no"
        )

        self.end_headers()

        collected_text = []

        try:
            while True:
                line = upstream.readline()

                if not line:
                    break

                text = line.decode(
                    "utf-8",
                    errors="replace"
                )

                if text.startswith("data: "):
                    payload = text[6:].strip()

                    if payload != "[DONE]":
                        try:
                            chunk = json.loads(
                                payload
                            )

                            choices = chunk.get(
                                "choices",
                                []
                            )

                            if choices:
                                delta = choices[0].get(
                                    "delta",
                                    {}
                                )

                                content = delta.get(
                                    "content"
                                )

                                if content:
                                    collected_text.append(
                                        content
                                    )

                        except json.JSONDecodeError:
                            pass

                    else:
                        call_writer(question)

                        text = "data: [DONE]\n\n"

                self.wfile.write(
                    text.encode("utf-8")
                )
                self.wfile.flush()

        except BrokenPipeError:
            print(
                "[Memory Proxy] Client disconnected."
            )

        finally:
            try:
                upstream.close()
            except Exception:
                pass


def main():
    print("================================")
    print("Memory Proxy")
    print("================================")
    print(
        f"Listening: http://{HOST}:{PORT}"
    )
    print(
        "Chat API:  "
        f"http://{HOST}:{PORT}/v1/chat/completions"
    )
    print(
        "Models:    "
        f"http://{HOST}:{PORT}/v1/models"
    )
    print("================================")
    print("Press Ctrl+C to stop.")

    server = ThreadingHTTPServer(
        (HOST, PORT),
        MemoryProxyHandler
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Memory Proxy...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
