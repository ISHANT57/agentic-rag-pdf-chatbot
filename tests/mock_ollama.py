"""A tiny stand-in for the Ollama HTTP API, used to smoke-test the notebook on machines
without a GPU. It returns schema-valid JSON and simple extractive answers, so every
code path of the agent, UI helpers and benchmark can run. It says nothing about
answer quality; run the real notebook on a Colab T4 for that.

Usage:
    python tests/mock_ollama.py            # listens on 127.0.0.1:11434
"""
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def fake_json(schema: dict, prompt: str) -> dict:
    props = schema.get("properties", {})
    last_msg = re.search(r"Latest user message: (.*)", prompt)
    question = last_msg.group(1).strip() if last_msg else ""
    out = {}
    for key, spec in props.items():
        if key == "route":
            q = question.lower()
            out[key] = "chitchat" if q.startswith(("hi", "hello", "thanks")) else (
                "summary" if "summar" in q else "document")
        elif key == "standalone_question":
            out[key] = question
        elif key == "relevant_ids":
            n = len(re.findall(r"^\[\d+\]", prompt, re.M))
            out[key] = [0, 1][: max(0, n)] if "France" not in prompt and "FIFA" not in prompt else []
        elif key == "grade":
            out[key] = "correct" if "NOT_IN_DOCUMENT" not in prompt else "partial"
        elif key == "claims":
            out[key] = [{"claim": "mock claim", "supported": True}]
        elif spec.get("type") == "boolean":
            out[key] = True
        elif spec.get("type") == "array":
            out[key] = []
        elif spec.get("type") in ("integer", "number"):
            out[key] = 1
        else:
            out[key] = "mock " + key
    return out


def fake_text(messages: list) -> str:
    prompt = "\n".join(m.get("content", "") for m in messages)
    if messages and messages[-1].get("images"):
        return "Figure 1: a line chart of monthly sales with a downward trend."
    if "Reply with exactly: ready" in prompt:
        return "ready"
    m = re.search(r"\[S1\][^\n]*\n(.+)", prompt)
    if "Sources:" in prompt and m:
        if re.search(r"France|FIFA", prompt):
            return "NOT_FOUND"
        sentence = re.split(r"(?<=[.!?])\s", m.group(1).strip())[0][:300]
        return f"{sentence} [S1]"
    return "This is a mock summary of the document."


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/api/version":
            self._send({"version": "mock-0.0"})
        elif self.path == "/api/tags":
            self._send({"models": [{"name": "mock:latest"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/api/pull":
            self._send({"status": "success"})
        elif self.path == "/api/generate":
            self._send({"done": True})
        elif self.path == "/api/chat":
            msgs = req.get("messages", [])
            prompt = "\n".join(m.get("content", "") for m in msgs)
            fmt = req.get("format")
            content = json.dumps(fake_json(fmt, prompt)) if isinstance(fmt, dict) else fake_text(msgs)
            self._send({"message": {"role": "assistant", "content": content}, "done": True,
                        "prompt_eval_count": len(prompt) // 4, "eval_count": len(content) // 4})
        else:
            self._send({"error": "not found"}, 404)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 11434), Handler).serve_forever()
