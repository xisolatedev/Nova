"""Run with python3 -m unittest discover -s tests -v; no weights required."""
import io
import json
import unittest
from unittest.mock import patch

from inference.backend import LlamaBackend, server_command


class BackendTests(unittest.TestCase):
    def test_cpu_dflash_command(self):
        command = server_command("target.gguf", "draft.gguf", threads=4, context=2048, block=7)
        for flag, value in (("--gpu-layers", "0"), ("--spec-draft-ngl", "0"),
                            ("--spec-type", "draft-dflash"), ("--spec-draft-n-max", "7"),
                            ("--host", "127.0.0.1")):
            self.assertEqual(command[command.index(flag) + 1], value)

    def test_reject_invalid_context_and_block(self):
        for options in ({"block": 16}, {"threads": -1}, {"context": 512}):
            with self.assertRaises(ValueError):
                server_command(**options)

    def test_sse_utf8_and_timing_counters(self):
        backend = object.__new__(LlamaBackend)
        backend.url = "http://127.0.0.1:1234"
        events = [{"choices": [{"delta": {"content": "café"}}]},
                  {"choices": [], "timings": {"draft_n": 7, "draft_n_accepted": 4}}]
        data = b": heartbeat\n\n" + b"".join(
            b"data: " + json.dumps(event, ensure_ascii=False).encode() + b"\n\n" for event in events)
        with patch("urllib.request.urlopen", return_value=io.BytesIO(data + b"data: [DONE]\n\n")):
            self.assertEqual(list(backend.stream([])), events)
            self.assertEqual(backend.last_timings["draft_n_accepted"], 4)

    def test_failed_initialization_closes_owned_process(self):
        with patch("pathlib.Path.is_file", return_value=True), \
             patch("subprocess.Popen") as popen:
            popen.return_value.poll.return_value = 1
            with self.assertRaisesRegex(RuntimeError, "exited"):
                LlamaBackend(port=19999)


if __name__ == "__main__":
    unittest.main()
