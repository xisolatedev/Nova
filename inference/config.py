"""Paths and conservative defaults for Linux CPUs and Codespaces."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLAMA_COMMIT = "1ba568c5ad5c6c718984bd2bf4e7c71485b706f9"
LLAMA = ROOT / ".runtime/llama.cpp"
BIN = LLAMA / "build/bin"
MODELS = ROOT / "models"
TARGET = MODELS / "Qwen3-4B"
DRAFT = MODELS / "Qwen3-4B-DFlash-b16"
TARGET_GGUF = MODELS / "Qwen3-4B-Q4_K_M.gguf"
DRAFT_GGUF = MODELS / "Qwen3-4B-DFlash-b16-Q8_0.gguf"


def hardware():
    """Respect affinity and cgroup CPU/memory limits, including Codespaces."""
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cores = min(cores, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    memory = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable"):
                memory[key] = int(value.split()[0]) * 1024
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit != "max":
            current = int(Path("/sys/fs/cgroup/memory.current").read_text())
            memory["MemTotal"] = min(memory["MemTotal"], int(limit))
            memory["MemAvailable"] = min(memory["MemAvailable"], max(0, int(limit) - current))
    except (OSError, ValueError):
        pass
    return {"cores": cores, "threads": min(6, cores), **memory}


def default_context():
    return 2048 if hardware().get("MemTotal", 0) < 12 * 1024**3 else 4096
