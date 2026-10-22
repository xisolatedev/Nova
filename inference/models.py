"""Resumable Hugging Face downloads, integrity checks and native conversion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .config import BIN, LLAMA, MODELS, TARGET, DRAFT, TARGET_GGUF, DRAFT_GGUF

REVISIONS = {
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "z-lab/Qwen3-4B-DFlash-b16": "b74e3a329c4d963783143b1e970d95b002be72bd",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(repo, directory):
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().model_info(repo, revision=REVISIONS[repo], files_metadata=True)
    selected = [f for f in info.siblings if f.rfilename.endswith(
        (".safetensors", ".json", "merges.txt", "vocab.txt", "tokenizer.model"))]
    if not any(f.rfilename.endswith(".safetensors") for f in selected):
        raise RuntimeError(f"No safetensors weights found for {repo}")
    manifest = {"repo": repo, "revision": info.sha, "files": {}}
    for item in selected:
        path = directory / item.rfilename
        expected_hash = item.lfs.sha256 if item.lfs else None

        def valid():
            return (path.is_file() and path.stat().st_size == item.size and
                    (not expected_hash or sha256(path) == expected_hash))

        if not valid():
            print(f"Downloading {repo}/{item.rfilename} at {info.sha}", flush=True)
            hf_hub_download(repo, item.rfilename, revision=info.sha, local_dir=directory,
                            force_download=path.exists())
        if not valid():
            raise RuntimeError(f"Size/SHA256 verification failed: {path}")
        manifest["files"][item.rfilename] = {"bytes": item.size, "sha256": sha256(path)}
    (directory / "nova-source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def validate_gguf(path, architecture):
    # The converter's own reader checks metadata and tensor table boundaries.
    sys.path.insert(0, str(LLAMA / "gguf-py"))
    from gguf import GGUFReader
    reader = GGUFReader(path)
    actual = reader.fields["general.architecture"].contents()
    if actual != architecture:
        raise ValueError(f"{path}: expected {architecture}, got {actual}")
    if not reader.tensors:
        raise ValueError(f"{path}: empty tensor table")
    for tensor in reader.tensors:
        if tensor.data_offset + tensor.n_bytes > Path(path).stat().st_size:
            raise ValueError(f"{path}: truncated tensor {tensor.name}")
    if architecture == "dflash":
        for key in ("dflash.block_size", "dflash.target_layers", "tokenizer.ggml.mask_token_id"):
            if key not in reader.fields:
                raise ValueError(f"DFlash missing {key}")


def convert(source, output, quant, architecture, target=None, keep_intermediate=False):
    if not (BIN / "llama-quantize").is_file():
        raise FileNotFoundError("Build llama-quantize first: run ./scripts/setup.sh")
    fingerprint = {
        "source": json.loads((source / "nova-source.json").read_text()),
        "llama_commit": subprocess.check_output(["git", "-C", str(LLAMA), "rev-parse", "HEAD"], text=True).strip(),
        "quant": quant,
    }
    if target:
        fingerprint["target"] = json.loads((target / "nova-source.json").read_text())
    stamp = Path(str(output) + ".json")
    if output.exists() and stamp.exists():
        previous = json.loads(stamp.read_text())
        if previous.get("inputs") == fingerprint and previous.get("sha256") == sha256(output):
            validate_gguf(output, architecture)
            print(f"Verified existing {output}", flush=True)
            return
    intermediate = MODELS / f"{source.name}-BF16.gguf"
    intermediate_stamp = Path(str(intermediate) + ".json")
    intermediate_inputs = {k: v for k, v in fingerprint.items() if k != "quant"}
    temporary = Path(str(output) + ".partial")
    # BF16 conversion uses lazy, mmap-backed safetensors rather than full FP32 loading.
    needed = sum(p.stat().st_size for p in source.glob("*.safetensors")) + 3 * 1024**3
    command = [sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(source),
               "--outtype", "bf16", "--outfile", str(intermediate) + ".partial"]
    if target:
        command += ["--target-model-dir", str(target)]
    reusable = (intermediate.is_file() and intermediate_stamp.is_file() and
                json.loads(intermediate_stamp.read_text()) == intermediate_inputs)
    if reusable:
        validate_gguf(intermediate, architecture)
    else:
        if shutil.disk_usage(MODELS).free < needed:
            raise RuntimeError(f"Conversion needs about {needed / 1024**3:.1f} GiB free in {MODELS}")
        subprocess.run(command, check=True)
        validate_gguf(str(intermediate) + ".partial", architecture)
        os.replace(str(intermediate) + ".partial", intermediate)
        intermediate_stamp.write_text(json.dumps(intermediate_inputs, indent=2) + "\n")
    subprocess.run([str(BIN / "llama-quantize"), str(intermediate), str(temporary), quant, "2"], check=True)
    validate_gguf(temporary, architecture)
    os.replace(temporary, output)
    stamp.write_text(json.dumps({"inputs": fingerprint, "sha256": sha256(output)}, indent=2) + "\n")
    if not keep_intermediate:
        intermediate.unlink()
        intermediate_stamp.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-quant", choices=["Q4_K_M", "Q5_K_M", "Q8_0"], default="Q4_K_M")
    parser.add_argument("--draft-quant", choices=["Q8_0", "Q4_K_M"], default="Q8_0")
    parser.add_argument("--keep-intermediate", action="store_true")
    args = parser.parse_args()
    MODELS.mkdir(exist_ok=True)
    download("Qwen/Qwen3-4B", TARGET)
    download("z-lab/Qwen3-4B-DFlash-b16", DRAFT)
    config = json.loads((DRAFT / "config.json").read_text())
    if not set(config.get("architectures", [])) & {"DFlashDraftModel", "DFlash2DraftModel"}:
        raise RuntimeError("The downloaded draft architecture is not supported by this setup")
    convert(TARGET, MODELS / f"Qwen3-4B-{args.target_quant}.gguf", args.target_quant, "qwen3",
            keep_intermediate=args.keep_intermediate)
    convert(DRAFT, MODELS / f"Qwen3-4B-DFlash-b16-{args.draft_quant}.gguf", args.draft_quant,
            "dflash", target=TARGET, keep_intermediate=args.keep_intermediate)


if __name__ == "__main__":
    main()
