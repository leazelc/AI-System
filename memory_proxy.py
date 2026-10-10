import os
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
from answer_evidence_gate import evaluate_evidence, refusal_message
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


def _memory_min_score():
    try:
        value = float(os.environ.get("AI_MEMORY_MIN_SCORE", "0.45"))
    except (TypeError, ValueError):
        value = 0.45
    return max(0.0, min(1.0, value))


def build_memory_context(question):
    """Build evidence-grounded context and reject unsupported memory answers."""
    try:
        memories = search(question, limit=10)
    except Exception as exc:
        print(f"[Memory Proxy] Memory retrieval failed: {exc}")
        memories = []

    decision = evaluate_evidence(
        question,
        memories,
        min_score=_memory_min_score(),
    )

    if not decision["answerable_from_retrieval"]:
        return (
            "【个人记忆证据门控】\n"
            "当前检索没有找到达到最低分数阈值的有效记忆。\n"
            "如果用户询问的是其个人经历、偏好、计划或历史信息，"
            "必须明确说明现有记忆证据不足，不得猜测或编造。\n"
            "如果问题是一般知识问题，可以正常回答，但不得把猜测说成用户的个人记忆。"
        )

    blocks = []
    for item in decision["evidence"]:
        blocks.append(
            f"[记忆证据 | score={item['score']:.3f}]\n"
            f"{item['content']}"
        )

    instructions = """【个人长期记忆问答规则】
仅将以下检索内容作为个人记忆证据。
证据没有明确支持的个人事实，不得自行补全或猜测。
若问题询问个人信息而证据不足，明确说明不知道。
一般知识问题仍可正常回答，但必须区分一般知识与个人记忆。
"""
    return instructions + "\n\n" + "\n\n".join(blocks)

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
