
import argparse
import time
import threading

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer
)

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--max-tokens", type=int, default=1024)
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
parser.add_argument("--quant", choices=["none", "q8"], default="q8")
args = parser.parse_args()

if args.threads < 1 or args.max_tokens < 1:
    parser.error("--threads and --max-tokens must be positive")

if args.quant == "q8" and args.device != "cpu":
    parser.error("PyTorch dynamic INT8 quantization requires CPU inference")

device = args.device

if device == "cpu":
    torch.set_num_threads(args.threads)

checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"

print(f"Loading {checkpoint} on {device}...")

load_start = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    dtype=torch.float32,
    attn_implementation="eager"
)
model.eval()
if args.quant == "q8":
    print("Applying dynamic INT8 quantization...")

    # Quantize supported Linear layers, including the language
    # model's attention and feed-forward projections.
    # Embeddings and other unsupported layers remain floating point.
    model = torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

if args.verbose:
    from collections import Counter

    dtypes = Counter(
        str(p.dtype)
        for p in model.parameters()
    )

    print(f"[DEBUG] Remaining parameter dtypes: {dict(dtypes)}")

model = model.to(device)

load_time = time.perf_counter() - load_start

if args.verbose:
    parameters = sum(p.numel() for p in model.parameters())

    quantized_layers = sum(
        isinstance(
            layer,
            torch.ao.nn.quantized.dynamic.Linear
        )
        for layer in model.modules()
    )

    print("\n[DEBUG] Model information")
    print(f"  Parameters (unpacked): {parameters:,}")
    print(f"  Device: {device}")
    print(f"  Quantization: {args.quant}")
    print(f"  Quantized Linear layers: {quantized_layers}")
    print(f"  CPU threads: {torch.get_num_threads()}")
    print(f"  Load time: {load_time:.2f}s")
    print(f"  Max new tokens: {args.max_tokens}")

messages = []

while True:
    prompt = input("\nYou: ").strip()

    if prompt.lower() in ["/quit", "/exit"]:
        break

    if not prompt:
        continue

    messages.append({
        "role": "user",
        "content": prompt
    })

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(device)

    input_tokens = inputs["input_ids"].shape[1]

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=30
    )

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": args.max_tokens,
        "temperature": 0.2,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id
    }

    result = {}
    errors = []

    def generate():
        try:
            with torch.inference_mode():
                result["output"] = model.generate(**generation_kwargs)
        except Exception as exc:
            errors.append(exc)
            streamer.end()

    thread = threading.Thread(target=generate)

    if args.verbose:
        print("\n[DEBUG] Generation settings")
        print(f"  Input tokens: {input_tokens}")
        print(f"  Max output tokens: {args.max_tokens}")
        print("  Temperature: 0.2")
        print("  Top-p: 0.9")

    print("\nAI: ", end="", flush=True)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    first_token_time = None
    response = ""

    thread.start()

    try:
        for text in streamer:
            if first_token_time is None and text:
                first_token_time = time.perf_counter()

            print(text, end="", flush=True)
            response += text
    finally:
        thread.join()

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    if errors:
        print(f"\n[ERROR] Generation failed: {errors[0]}")
        messages.pop()
        continue

    output = result["output"]
    output_tokens = output.shape[1] - input_tokens

    elapsed = end - start
    ttft = (
        first_token_time - start
        if first_token_time is not None else None
    )

    overall_tps = output_tokens / elapsed if elapsed > 0 else 0

    decode_time = (
        end - first_token_time
        if first_token_time is not None else 0
    )
    decode_tokens = max(output_tokens - 1, 0)
    decode_tps = (
        decode_tokens / decode_time
        if decode_time > 0 else 0
    )

    messages.append({
        "role": "assistant",
        "content": response
    })

    print(f"\n\n[Speed] {overall_tps:.2f} tokens/sec")

    if args.verbose:
        print("\n[DEBUG] Performance")
        print(f"  Input tokens: {input_tokens}")
        print(f"  Output tokens: {output_tokens}")
        print(f"  Total time: {elapsed:.3f}s")
        print(
            f"  Time to first visible text: {ttft:.3f}s"
            if ttft is not None else
            "  Time to first visible text: N/A"
        )
        print(f"  Overall throughput: {overall_tps:.2f} tokens/sec")
        print(f"  Approx. decode speed: {decode_tps:.2f} tokens/sec")

        if device == "cuda":
            print(
                f"  GPU allocated: "
                f"{torch.cuda.memory_allocated() / 1024**2:.1f} MiB"
            )
            print(
                f"  GPU peak allocated: "
                f"{torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB"
            )
