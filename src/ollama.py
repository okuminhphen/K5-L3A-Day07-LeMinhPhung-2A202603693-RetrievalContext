"""Structured chat requests to the local Ollama service."""
import json
from urllib.request import Request, urlopen


DEFAULT_MODEL = "llama3.2:latest"


def chat_json(
    model: str, messages: list[dict[str, str]], schema: dict, *,
    usage: list[dict[str, int | None]] | None = None,
) -> dict:
    request = Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps({
            "model": model,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0, "num_predict": 256},
            "messages": messages,
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120) as response:
        payload = json.load(response)
    if usage is not None:
        usage.append({
            "prompt_tokens": payload.get("prompt_eval_count"),
            "output_tokens": payload.get("eval_count"),
        })
    return json.loads(payload["message"]["content"])
