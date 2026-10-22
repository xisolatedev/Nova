#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Nova setup failed at line $LINENO. Fix the reported error, then rerun; downloads resume." >&2' ERR
cd "$(dirname "$0")/.."
LLAMA_COMMIT=1ba568c5ad5c6c718984bd2bf4e7c71485b706f9
export PIP_CACHE_DIR="$PWD/.runtime/pip-cache"
export HF_HOME="$PWD/.runtime/huggingface"
if [[ $(uname -s) != Linux ]]; then
    echo 'This CPU setup supports Linux, including Codespaces.' >&2
    exit 1
fi
missing=0
for tool in git cmake g++ make python3; do
    if ! command -v "$tool" >/dev/null; then missing=1; fi
done
if (( missing )) || ! python3 -m venv --help >/dev/null 2>&1; then
    if command -v apt-get >/dev/null && command -v sudo >/dev/null; then
        sudo apt-get update
        sudo apt-get install -y build-essential cmake git python3-venv libssl-dev
    else
        echo 'Install build-essential, cmake, git, python3-venv and libssl-dev using your package manager.' >&2
        exit 1
    fi
fi
mkdir -p .runtime models
if [[ ! -d .runtime/llama.cpp/.git ]]; then
    git init .runtime/llama.cpp
    git -C .runtime/llama.cpp remote add origin https://github.com/ggml-org/llama.cpp
fi
if [[ $(git -C .runtime/llama.cpp rev-parse HEAD 2>/dev/null || true) != "$LLAMA_COMMIT" ]]; then
    git -C .runtime/llama.cpp fetch --depth 1 origin "$LLAMA_COMMIT"
    git -C .runtime/llama.cpp checkout --detach "$LLAMA_COMMIT"
fi
python3 -m venv .runtime/venv
.runtime/venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
.runtime/venv/bin/python -m pip install -r .runtime/llama.cpp/requirements/requirements-convert_legacy_llama.txt 'huggingface_hub>=0.34,<1.0'
.runtime/venv/bin/python -c 'import torch; assert torch.version.cuda is None, "Expected CPU-only PyTorch"'
threads=$(python3 -c 'from inference.config import hardware; print(min(2, hardware()["threads"]))')
python3 -c 'from inference.config import hardware; print("CPU and RAM:", hardware())'
cmake -S .runtime/llama.cpp -B .runtime/llama.cpp/build \
    -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CCACHE=OFF \
    -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_HIP=OFF -DGGML_METAL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON
cmake --build .runtime/llama.cpp/build --target llama-server llama-quantize -j "$threads"
.runtime/venv/bin/python -m inference.models "$@"
target_quant=Q4_K_M
draft_quant=Q8_0
while (( $# )); do
    case "$1" in
        --target-quant) target_quant="$2"; shift 2 ;;
        --draft-quant) draft_quant="$2"; shift 2 ;;
        *) shift ;;
    esac
done
python3 -m inference.cli --model "models/Qwen3-4B-${target_quant}.gguf" --prompt 'Say hello in one sentence.' --max-tokens 24 --temperature 0
python3 -m inference.cli --model "models/Qwen3-4B-${target_quant}.gguf" --draft "models/Qwen3-4B-DFlash-b16-${draft_quant}.gguf" --dflash --prompt 'What is two plus two?' --max-tokens 24 --temperature 0
echo 'Nova setup passed baseline and target-verified DFlash inference.'
