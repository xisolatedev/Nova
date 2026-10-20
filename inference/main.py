import torch
from transformers import AutoTokenizer, AutoModel

model_id = "z-lab/Qwen3-4B-DFlash-b16"

print("🤖 Loading Qwen3 tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-4B"
)

print("💾 Loading DFlash model into CPU RAM...")

model = AutoModel.from_pretrained(
    model_id,
    dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True
)

model.eval()

print("✅ Model loaded successfully!")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Device: {next(model.parameters()).device}")
print(f"Dtype: {next(model.parameters()).dtype}")

print("⚙️ Testing tokenization...")

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

print(f"Input token IDs: {inputs['input_ids'].tolist()}")
print("✅ Tokenization successful!")
print("✅ DFlash draft model ready.")
print("Note: Text generation requires the Qwen3-4B target model and DFlash decoding implementation.")
