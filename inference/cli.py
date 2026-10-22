"""Terminal chat and loopback OpenAI-compatible HTTP service."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from .backend import LlamaBackend, server_command
from .config import TARGET_GGUF, DRAFT_GGUF, hardware, default_context


def main(default_dflash=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=TARGET_GGUF)
    parser.add_argument("--draft", type=Path, default=DRAFT_GGUF)
    parser.add_argument("--dflash", action="store_true", default=default_dflash)
    parser.add_argument("--threads", type=int, default=hardware()["threads"])
    parser.add_argument("--context", type=int, default=default_context())
    parser.add_argument("--block", type=int, default=15)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", help="Run one prompt and exit")
    parser.add_argument("--serve", action="store_true")3
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    draft = args.draft if args.dflash else None
    try:
        if args.serve:
            command = server_command(args.model, draft, args.threads, args.context,
                                     args.block, args.port, args.host)
            print(f"Nova API: http://{args.host}:{args.port}/v1/chat/completions", flush=True)
            env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}
            os.execve(command[0], command, env)
        print(f"Loading {args.model.name}; threads={args.threads}, context={args.context}, "
              f"DFlash={bool(draft)}", file=sys.stderr, flush=True)
        with LlamaBackend(args.model, draft, args.threads, args.context, args.block) as backend:
            messages = []
            print("Ready. /reset clears history; /quit exits. Use --model to switch models.", file=sys.stderr)
            while True:
                prompt = args.prompt if args.prompt is not None else input("\nYou: ")
                if prompt.strip().lower() in {"/quit", "exit", "quit"}:
                    break
                if prompt.strip() == "/reset":
                    messages.clear()
                    continue
                messages.append({"role": "user", "content": prompt})
                content = []
                started, first = time.perf_counter(), None
                print("Nova: ", end="", flush=True)
                for event in backend.stream(messages, args.max_tokens, args.temperature, args.top_p, args.seed):
                    for choice in event.get("choices", []):
                        token = choice.get("delta", {}).get("content") or ""
                        if token:
                            first = first or time.perf_counter()
                            content.append(token)
                            print(token, end="", flush=True)
                print()
                messages.append({"role": "assistant", "content": "".join(content)})
                print(json.dumps({"ttft_seconds": first - started if first else None,
                                  "timings": backend.last_timings,
                                  "memory_MiB": backend.memory()}), file=sys.stderr)
                if args.prompt is not None:
                    break
    except (KeyboardInterrupt, EOFError):
        pass
    except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
        parser.exit(1, f"Nova: {exc}\n")


if __name__ == "__main__":
    main()
