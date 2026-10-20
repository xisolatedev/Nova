from transformers import MossTTSRealtime
import torch
import soundfile as sf

model = MossTTSRealtime.from_pretrained("./models/MOSS-TTS-Realtime", dtype="auto")

text = "Hello from Raspberry Pi!"
with torch.no_grad():
    audio = model.tts(text)  # returns waveform tensor

sf.write("output.wav", audio.cpu().numpy(), samplerate=22050)
print("✅ Audio saved")
