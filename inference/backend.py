"""Persistent llama-server process and a dependency-free streaming client."""
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request

from .config import BIN, ROOT, TARGET_GGUF, DRAFT_GGUF, default_context, hardware


def read_json(url, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def server_command(model=TARGET_GGUF, draft=None, threads=None, context=None,
                   block=15, port=8080, host="127.0.0.1"):
    threads = threads or hardware()["threads"]
    context = context or default_context()
    if threads < 1 or context < 256 or not 1 <= block <= 15:
        raise ValueError("Require threads >= 1, context >= 256, draft block 1..15")
    command = [str(BIN / "llama-server"), "--model", str(model), "--host", host,
               "--port", str(port), "--threads", str(threads), "--threads-batch", str(threads),
               "--ctx-size", str(context), "--parallel", "1", "--gpu-layers", "0",
               "--batch-size", "256", "--ubatch-size", "256", "--flash-attn", "on",
               "--jinja", "--reasoning-budget", "0", "--metrics"]
    if draft:
        command += ["--spec-draft-model", str(draft), "--spec-type", "draft-dflash",
                    "--spec-draft-n-max", str(block), "--spec-draft-ngl", "0",
                    "--spec-draft-threads", str(threads), "--spec-draft-threads-batch", str(threads)]
    return command


class LlamaBackend:
    """Reuse loaded weights across calls; close or use as a context manager."""

    def __init__(self, model=TARGET_GGUF, draft=None, threads=None, context=None,
                 block=15, port=0, startup_timeout=600):
        self.model, self.draft = Path(model), Path(draft) if draft else None
        self.process = None
        self.log = None
        self.last_timings = {}
        for path in (BIN / "llama-server", self.model, self.draft):
            if path is not None and not path.is_file():
                raise FileNotFoundError(f"Missing {path}; run ./scripts/setup.sh")
        if not port:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.command = server_command(self.model, self.draft, threads, context, block, port)
        log_dir = ROOT / ".runtime/logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"server-{port}.log"
        self.log = self.log_path.open("w")
        try:
            # Prevent environment variables from silently overriding CPU/model settings.
            env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}
            self.process = subprocess.Popen(self.command, stdout=self.log, stderr=self.log, env=env)
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"llama-server exited {self.process.returncode}; see {self.log_path}")
                try:
                    health = read_json(self.url + "/health", timeout=1)
                    if health.get("status") == "ok":
                        return
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(0.2)
            raise TimeoutError(f"Model loading timed out; see {self.log_path}")
        except BaseException:
            self.close()
            raise

    def stream(self, messages, max_tokens=512, temperature=0.7, top_p=0.8, seed=42,
               cache_prompt=True, ignore_eos=False):
        """Yield OpenAI SSE events, retaining llama.cpp timing counters."""
        self.last_timings = {}
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "top_p": top_p, "seed": seed, "stream": True,
                "stream_options": {"include_usage": True}, "timings_per_token": True,
                "cache_prompt": cache_prompt, "ignore_eos": ignore_eos}
        request = urllib.request.Request(self.url + "/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                if "error" in event:
                    raise RuntimeError(event["error"])
                if "timings" in event:
                    self.last_timings = event["timings"]
                yield event

    def memory(self):
        """Linux process RSS and high-water mark in MiB (includes loading)."""
        values = {}
        if self.process:
            try:
                for line in Path(f"/proc/{self.process.pid}/status").read_text().splitlines():
                    if line.startswith(("VmRSS:", "VmHWM:")):
                        key, value = line.split(":", 1)
                        values[key] = int(value.split()[0]) / 1024
            except OSError:
                pass
        return values

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log:
            self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
