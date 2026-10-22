"""Live regression checks: target verification, chat history, streaming and API."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.backend import LlamaBackend, read_json
from inference.config import DRAFT_GGUF


def collect(backend, messages):
    output = []
    for event in backend.stream(messages, max_tokens=24, temperature=0):
        for choice in event.get("choices", []):
            output.append(choice.get("delta", {}).get("content") or "")
    text = "".join(output)
    assert text.strip(), "No visible streamed response"
    assert backend.last_timings.get("predicted_n", 0) > 0, "Missing timing counters"
    return text


def main():
    results = []
    for draft in (None, DRAFT_GGUF):
        with LlamaBackend(draft=draft, context=2048, block=7) as backend:
            pid = backend.process.pid
            messages = [{"role": "user", "content": "Reply with only the number: what is 2 + 2?"}]
            text = collect(backend, messages)
            assert "4" in text, text
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": "Add three to that number. Reply with only the number."}]
            followup = collect(backend, messages)
            assert "7" in followup, followup
            assert backend.process.pid == pid, "Model process was unnecessarily reloaded"
            reply = read_json(backend.url + "/v1/chat/completions", {
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 24, "temperature": 0, "stream": False})
            assert reply["choices"][0]["message"]["content"].strip(), reply
            models = read_json(backend.url + "/v1/models")
            assert models["data"], models
            results.append({"dflash": bool(draft), "arithmetic": text, "followup": followup,
                            "timings": backend.last_timings, "memory_MiB": backend.memory()})
        assert backend.process.poll() is not None, "Server process leaked"
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
