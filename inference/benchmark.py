"""Measure baseline and DFlash on identical, uncached, greedy requests."""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

from .backend import LlamaBackend
from .config import TARGET_GGUF, DRAFT_GGUF, ROOT, LLAMA, hardware

PROMPTS = [
    "Explain why the sky appears blue in simple terms.",
    "Write a Python function to find prime numbers and explain how it works.",
    "Describe the steps for growing tomatoes from seed.",
]


def measure(backend, prompt, tokens):
    start = time.perf_counter()
    first = None
    output = []
    usage = {}
    for event in backend.stream([{"role": "user", "content": prompt}], max_tokens=tokens,
                                temperature=0, seed=42, cache_prompt=False, ignore_eos=True):
        usage = event.get("usage", usage)
        for choice in event.get("choices", []):
            token = choice.get("delta", {}).get("content") or ""
            if token:
                first = first or time.perf_counter()
                output.append(token)
    elapsed = time.perf_counter() - start
    timings = backend.last_timings.copy()
    generated = usage.get("completion_tokens", timings.get("predicted_n", 0))
    if not generated or not output or first is None:
        raise RuntimeError("Inference returned no visible output or token counts")
    drafted = timings.get("draft_n", 0)
    return {"prompt": prompt, "text": "".join(output), "tokens": generated,
            "elapsed_seconds": elapsed, "ttft_seconds": first - start,
            "end_to_end_tps": generated / elapsed, "timings": timings,
            "acceptance_rate": timings.get("draft_n_accepted", 0) / drafted if drafted else None,
            "memory_MiB": backend.memory()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, nargs="+", default=[TARGET_GGUF])
    parser.add_argument("--drafts", type=Path, nargs="+", default=[DRAFT_GGUF])
    parser.add_argument("--threads", type=int, nargs="+", default=sorted({max(1, hardware()["threads"] // 2), hardware()["threads"]}))
    parser.add_argument("--blocks", type=int, nargs="+", default=[3, 7, 15])
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/latest.json")
    args = parser.parse_args()
    if args.tokens < 1 or args.repeats < 1 or any(t < 1 for t in args.threads):
        parser.error("tokens, repeats and threads must be positive")
    for path in args.models + args.drafts:
        if not path.is_file():
            parser.error(f"Missing model: {path}; run setup for this quantization first")
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "hardware": hardware(),
              "llama_commit": subprocess.check_output(["git", "-C", str(LLAMA), "rev-parse", "HEAD"], text=True).strip(),
              "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ("models", "drafts")},
              "runs": [], "errors": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".partial")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    configurations = []
    for model, threads in itertools.product(args.models, args.threads):
        configurations.append((model, threads, None, 15))
        configurations.extend((model, threads, draft, block) for draft, block in itertools.product(args.drafts, args.blocks))
    for model, threads, draft, block in configurations:
        label = {"model": str(model), "threads": threads, "draft": str(draft) if draft else None,
                 "block": block if draft else None, "context": args.context}
        print(json.dumps(label), flush=True)
        try:
            with LlamaBackend(model, draft, threads, args.context, block) as backend:
                measure(backend, "Say hello.", 4)
                samples = [measure(backend, PROMPTS[i % len(PROMPTS)], args.tokens) for i in range(args.repeats)]
                result = {**label, "samples": samples,
                          "median_end_to_end_tps": statistics.median(s["end_to_end_tps"] for s in samples),
                          "median_ttft_seconds": statistics.median(s["ttft_seconds"] for s in samples),
                          "peak_process_RSS_MiB": max(s["memory_MiB"].get("VmHWM", 0) for s in samples),
                          "command": backend.command, "log": str(backend.log_path)}
                report["runs"].append(result)
                print(f"  {result['median_end_to_end_tps']:.3f} tok/s, TTFT {result['median_ttft_seconds']:.3f}s", flush=True)
        except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
            report["errors"].append({**label, "error": str(exc)})
            print(f"  FAILED: {exc}", flush=True)
        save()
    for run in report["runs"]:
        baselines = [r for r in report["runs"] if not r["draft"] and r["model"] == run["model"]]
        if baselines:
            baseline = max(baselines, key=lambda r: r["median_end_to_end_tps"])
            run["speedup_vs_best_baseline"] = run["median_end_to_end_tps"] / baseline["median_end_to_end_tps"]
    if report["runs"]:
        winner = max(report["runs"], key=lambda r: r["median_end_to_end_tps"])
        report["recommendation"] = {k: winner[k] for k in ("model", "draft", "threads", "block", "median_end_to_end_tps")}
        print("Measured recommendation: " + json.dumps(report["recommendation"]), flush=True)
    save()
    if report["errors"] or not report["runs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
