import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = Path(__file__).resolve().parent.parent / "models" / "Qwen3-4B"

print("🤖 Loading Qwen3-4B...")

tokenizer = AutoTokenizer.from_pretrained(str(model_path))

model = AutoModelForCausalLM.from_pretrained(
    str(model_path),
    dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
).eval()

print("✅ Model loaded!")

while True:
    prompt = input("\nYou: ")

    if prompt.lower() in ("exit", "quit"):
        break

    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(text, return_tensors="pt")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            pad_token_id=tokenizer.eos_token_id
        )

    response = output[0][inputs["input_ids"].shape[1]:]

    print("\nNova:", tokenizer.decode(response, skip_special_tokens=True))