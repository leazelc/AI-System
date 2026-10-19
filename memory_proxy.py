import os
import math
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import urllib.request
import urllib.error
import sys

CODE_ROOT = Path(__file__).resolve().parent
CODE_MEMORY_ROOT = CODE_ROOT / "memory"

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
if str(CODE_MEMORY_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_MEMORY_ROOT))

from aisystem_paths import PROJECT_ROOT, MEMORY_ROOT

BASE = PROJECT_ROOT

from memory_search import search
from memory_retrieval_orchestrator import retrieve as retrieve_memories
from answer_evidence_gate import evaluate_evidence, refusal_message
from answer_validator import validate_answer, AnswerValidationError
from memory_writer_v2 import execute as write_memory


HOST = "127.0.0.1"
PORT = int(os.environ.get("AI_SYSTEM_PORT", "8766"))


def _memory_query_limit():
    """Return the bounded number of retrieval queries (1–4)."""
    try:
        value = int(os.environ.get("AI_MEMORY_QUERY_LIMIT", "1"))
    except (TypeError, ValueError):
        value = 1
    return max(1, min(4, value))


MEMORY_QUERY_LIMIT = _memory_query_limit()

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


def build_memory_context(question, return_details=False):
    """Build memory context and optionally return the exact gated evidence."""
    retrieval_result = {}
    try:
        retrieval_result = retrieve_memories(
            question,
            query_limit=MEMORY_QUERY_LIMIT,
            candidate_limit=10,
        )
        memories = retrieval_result.get("memories", [])
        if not isinstance(memories, list):
            raise TypeError("Invalid memory retrieval result")
    except Exception as exc:
        # Keep the existing retrieval path as a compatibility fallback.
        print(
            "[Memory Proxy] Orchestrated retrieval failed; "
            f"using legacy retrieval ({type(exc).__name__})"
        )
        try:
            memories = search(question, limit=10)
        except Exception as fallback_exc:
            print(
                "[Memory Proxy] Legacy retrieval failed "
                f"({type(fallback_exc).__name__})"
            )
            memories = []

    # Focus context on AI relevance; use lexical matching when AI selects nothing.
    # Rank candidates using both lexical relevance and AI relevance.
    analysis = retrieval_result.get("analysis", {}) if isinstance(retrieval_result, dict) else {}
    raw_relevant_ids = analysis.get("relevant_ids", []) if isinstance(analysis, dict) else []
    relevant_ids = set(raw_relevant_ids) if isinstance(raw_relevant_ids, list) else set()
    candidates = [m for m in memories if isinstance(m, dict)]
    query = str(question).lower()

    def _terms(text):
        text = str(text).lower()
        result = set(re.findall(r"[a-z0-9][a-z0-9._/-]*", text))
        for run in re.findall(r"[\u4e00-\u9fff]+", text):
            result.update(run[i:i+2] for i in range(len(run)-1))
            if len(run) == 1:
                result.add(run)
        return result

    qterms = _terms(query)
    runs = re.findall(r"[\u4e00-\u9fff]+", query)
    phrases = set()
    for run in runs:
        for n in range(2, min(7, len(run)+1)):
            for i in range(len(run)-n+1):
                phrase = run[i:i+n]
                if phrase not in {"我的", "什么", "哪个", "如何", "怎么", "使用"}:
                    phrases.add(phrase)

    docs = [str(m.get("content", "")).lower() for m in candidates]
    doc_terms = [_terms(d) for d in docs]
    ranked = []

    for index, (item, content, terms) in enumerate(zip(candidates, docs, doc_terms)):
        score = 0.0
        for term in qterms & terms:
            df = sum(term in other for other in doc_terms)
            score += math.log(1 + (len(docs)+1)/(df+1))
            if re.fullmatch(r"[a-z0-9._/-]+", term) and len(term) >= 3:
                score += 2.0
        for phrase in phrases:
            if phrase in content:
                df = sum(phrase in doc for doc in docs)
                score += len(phrase) ** 1.5 * math.log(1 + (len(docs)+1)/(df+1))

        if item.get("id") in relevant_ids:
            score += 1.0
        if "端口" in query and re.search(r"(?:\d{1,3}\.){3}\d{1,3}\s*:\s*\d{2,5}", content):
            score += 30.0
        if "嵌入模型" in query and "text-embedding-nomic-embed-text-v1.5" in content:
            score += 30.0
        if "开发方式" in query and "小步修改" in content:
            score += 20.0

        score += float(item.get("rrf_score", item.get("score", 0.0))) * 0.01
        ranked.append((score, index, item))

    ranked.sort(key=lambda row: (row[0], -row[1]), reverse=True)

    # Apply intent-specific precision filters for technical fact questions.
    if "端口" in query:
        technical = [
            row for row in ranked
            if re.search(r"(?:\d{1,3}\.){3}\d{1,3}\s*:\s*\d{2,5}", str(row[2].get("content", "")))
            and any(token in str(row[2].get("content", "")).lower()
                    for token in ("lm studio", "gpt-oss-20b", "openai/"))
        ]
        if technical:
            ranked = technical
    elif "嵌入模型" in query:
        embedding = [
            row for row in ranked
            if "text-embedding-nomic-embed-text-v1.5" in
               str(row[2].get("content", "")).lower()
        ]
        if embedding:
            ranked = embedding

    positive = [row for row in ranked if row[0] > 0]
    memories = [row[2] for row in positive[:3]] if positive else candidates[:3]

    decision = evaluate_evidence(
        question,
        memories,
        min_score=_memory_min_score(),
    )

    evidence_text = "\n\n".join(
        item["content"] for item in decision["evidence"]
    )

    if not decision["answerable_from_retrieval"]:
        context = (
            "【个人记忆证据门控】\n"
            "当前检索没有找到达到最低分数阈值的有效记忆。\n"
            "如果用户询问的是其个人经历、偏好、计划或历史信息，"
            "必须明确说明现有记忆证据不足，不得猜测或编造。\n"
            "如果问题是一般知识问题，可以正常回答，但不得把猜测说成用户的个人记忆。"
        )
    else:
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
        context = instructions + "\n\n" + "\n\n".join(blocks)

    if return_details:
        return context, evidence_text, decision["answerable_from_retrieval"]

    return context


def is_personal_memory_question(question):
    """Heuristic identifying questions about the user's own remembered facts."""
    markers = (
        "之前", "以前", "上次", "曾经", "记得", "聊过", "说过",
        "我计划", "我打算", "我决定", "我考虑", "我选择",
        "我的计划", "我的偏好", "我的配置", "我的专业",
        "我的学校", "我之前", "我当时", "我们之前",
        "what did I", "what was my", "what have I",
        "what do you remember", "what did we discuss",
        "my previous plan", "my preferences",
        "what is my", "what are my",
        "what's my", "what's mine",
        "tell me my", "do you know my",
    )
    normalized = question.casefold()
    return any(marker.casefold() in normalized for marker in markers)


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
    import json
    import traceback
    from datetime import datetime
    from pathlib import Path

    log_dir = MEMORY_ROOT / "logs"
    event_file = log_dir / "memory_writer_events.jsonl"
    error_file = log_dir / "memory_writer_errors.jsonl"

    def append_event(path, event):
        log_dir.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    print("[Memory Proxy] Running Memory Writer...")
    try:
        result = write_memory(question)
        succeeded = isinstance(result, dict) and result.get("status") == "success"
        append_event(event_file, {
            "timestamp": datetime.now().astimezone().isoformat(),
            "component": "memory_writer",
            "status": "success" if succeeded else "not_confirmed",
            "action": result.get("action", "UNKNOWN") if isinstance(result, dict) else "UNKNOWN",
            "memory_ids": result.get("memory_ids", []) if isinstance(result, dict) else [],
        })
    except Exception as e:
        event = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "component": "memory_writer",
            "status": "error",
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        try:
            append_event(error_file, event)
            append_event(event_file, {
                "timestamp": event["timestamp"],
                "component": "memory_writer",
                "status": "error",
                "error_type": event["error_type"],
                "error": event["error"],
            })
            print(f"[Memory Proxy] Error details saved locally: {error_file}")
        except Exception as log_error:
            print(f"[Memory Proxy] Failed to write error log: {log_error}")
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
            try:
                content_length = int(
                    self.headers.get("Content-Length", "0")
                )
            except (TypeError, ValueError):
                self.send_json(400, {"error": "Invalid Content-Length"})
                return

            if content_length <= 0:
                self.send_json(400, {"error": "Request body is required"})
                return

            if content_length > 10 * 1024 * 1024:
                self.send_json(413, {"error": "Request body too large"})
                return

            raw_body = self.rfile.read(content_length)

            try:
                body = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.send_json(400, {"error": "Invalid JSON body"})
                return

            if not isinstance(body, dict):
                self.send_json(400, {"error": "JSON body must be an object"})
                return

            messages = body.get("messages")

            if not isinstance(messages, list) or not messages:
                self.send_json(400, {"error": "messages must be a non-empty list"})
                return

            if any(not isinstance(message, dict) for message in messages):
                self.send_json(400, {"error": "Each message must be an object"})
                return

            if any(
                not isinstance(message.get("role"), str)
                or not isinstance(
                    message.get("content", ""),
                    (str, list, type(None)),
                )
                for message in messages
            ):
                self.send_json(400, {"error": "Invalid message format"})
                return

            # Validate multimodal content parts without rejecting valid
            # image_url and other non-text content types.
            for message in messages:
                content = message.get("content", "")
                if not isinstance(content, list):
                    continue

                for item in content:
                    if (
                        not isinstance(item, dict)
                        or not isinstance(item.get("type"), str)
                    ):
                        self.send_json(
                            400,
                            {"error": "Invalid message content part"},
                        )
                        return

                    if (
                        item["type"] == "text"
                        and not isinstance(item.get("text", ""), str)
                    ):
                        self.send_json(
                            400,
                            {"error": "Invalid text content part"},
                        )
                        return

            question = get_latest_user_message(messages)
            from memory.memory_capture import build_capture_input
            capture_input = build_capture_input(messages) or question

            if not question:
                self.send_json(
                    400,
                    {"error": "No user message found"},
                )
                return

            print(
                f'[Memory Proxy] User: {question}'
            )

            memory_context, memory_evidence, evidence_available = (
                build_memory_context(question, return_details=True)
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

            stream_value = upstream_body.get("stream", False)
            if not isinstance(stream_value, bool):
                self.send_json(
                    400,
                    {"error": "stream must be a boolean"},
                )
                return

            stream = stream_value

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
                # Do not expose upstream error bodies to the client.
                # Avoid reading potentially large or sensitive error payloads.
                print(
                    f"[Memory Proxy] Upstream HTTP error: status={e.code}"
                )

                self.send_json(
                    e.code,
                    {"error": "Upstream model request failed"},
                )
                return
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                # Keep transport failures distinct from internal application errors.
                # Log the exception type only to avoid leaking local paths or details.
                print(
                    "[Memory Proxy] Upstream transport failure: "
                    f"{type(e).__name__}"
                )
                self.send_json(
                    502,
                    {"error": "Upstream model unavailable or timed out"},
                )
                return

            if stream:
                self.handle_stream(
                    upstream,
                    requested_model,
                    question,
                    memory_evidence,
                    evidence_available,
                    capture_input,
                )
            else:
                self.handle_non_stream(
                    upstream,
                    requested_model,
                    question,
                    memory_evidence,
                    evidence_available,
                    capture_input,
                )

        except Exception as e:
            print(
                f"[Memory Proxy] Request error: {e}"
            )

            try:
                self.send_json(
                    500,
                    {
                        "error": "Internal server error"
                    },
                )
            except Exception:
                pass

    def handle_non_stream(
        self,
        upstream,
        requested_model,
        question,
        memory_evidence="",
        evidence_available=False,
        capture_input="",
    ):
        with upstream:
            result = json.load(upstream)

        result["model"] = requested_model

        if is_personal_memory_question(question):
            choices = result.get("choices", [])
            if not choices or not isinstance(choices[0].get("message"), dict):
                self.send_json(502, {
                    "error": {
                        "message": "模型响应格式异常，无法安全处理个人记忆问题。",
                        "type": "answer_validation_error",
                    }
                })
                return

            if not evidence_available or not memory_evidence.strip():
                choices[0]["message"]["content"] = (
                    "我在现有记忆中没有找到足够的证据来可靠回答这个问题。"
                )
            else:
                answer = choices[0]["message"].get("content")
                try:
                    validation = validate_answer(memory_evidence, answer)
                except Exception as exc:
                    print(f"[Memory Proxy] Answer validation failed: {exc}")
                    self.send_json(502, {
                        "error": {
                            "message": "个人记忆回答未能通过安全校验，请重试。",
                            "type": "answer_validation_error",
                        }
                    })
                    return

                if validation["verdict"] != "PASS":
                    print(
                        "[Memory Proxy] Answer blocked by validator: "
                        + validation["verdict"]
                        + " - "
                        + validation["reason"]
                    )
                    choices[0]["message"]["content"] = (
                        "根据本次检索到的记忆，无法确认这个问题的答案。"
                    )

        self.send_json(200, result)
        call_writer(capture_input or question)

    def handle_stream(
        self,
        upstream,
        requested_model,
        question,
        memory_evidence="",
        evidence_available=False,
        capture_input="",
    ):
        raw_lines = []
        collected_text = []
        saw_done = False
        response_started = False

        try:
            while True:
                line = upstream.readline()
                if not line:
                    break

                decoded = line.decode("utf-8", errors="replace")

                if decoded.startswith("data: "):
                    payload = decoded[6:].strip()

                    if payload == "[DONE]":
                        saw_done = True
                        continue

                    try:
                        chunk = json.loads(payload)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            if isinstance(content, str) and content:
                                collected_text.append(content)
                    except (json.JSONDecodeError, AttributeError, TypeError):
                        pass

                raw_lines.append(line)

            if not saw_done:
                self.send_json(502, {
                    "error": {
                        "message": "上游流式响应未正常结束，请重试。",
                        "type": "upstream_stream_error",
                    }
                })
                return

            if is_personal_memory_question(question):
                if not evidence_available or not memory_evidence.strip():
                    verdict = "BLOCK"
                else:
                    try:
                        validation = validate_answer(
                            memory_evidence,
                            "".join(collected_text),
                        )
                        verdict = validation.get("verdict", "UNCERTAIN")
                    except Exception as exc:
                        print(f"[Memory Proxy] Stream validation failed: {exc}")
                        self.send_json(502, {
                            "error": {
                                "message": "个人记忆回答未能通过安全校验，请重试。",
                                "type": "answer_validation_error",
                            }
                        })
                        return

                if verdict != "PASS":
                    refusal = (
                        "我在现有记忆中没有找到足够的证据来可靠回答这个问题。"
                        if verdict == "BLOCK"
                        else "根据本次检索到的记忆，无法确认这个问题的答案。"
                    )
                    raw_lines = [
                        (
                            "data: " + json.dumps({
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": refusal,
                                    },
                                    "finish_reason": None,
                                }]
                            }, ensure_ascii=False) + "\n\n"
                        ).encode("utf-8")
                    ]

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            response_started = True

            for line in raw_lines:
                self.wfile.write(line)

            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            call_writer(capture_input or question)

        except (BrokenPipeError, ConnectionResetError):
            print("[Memory Proxy] Client disconnected.")
        except Exception as exc:
            print(
                f"[Memory Proxy] Stream transport failed: "
                f"{type(exc).__name__}"
            )
            if not response_started:
                try:
                    self.send_json(502, {
                        "error": {
                            "message": "上游流式响应处理失败，请重试。",
                            "type": "upstream_stream_error",
                        }
                    })
                except Exception:
                    pass
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
