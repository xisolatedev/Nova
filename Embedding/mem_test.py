#!/usr/bin/env python3
# File: embedding_test.py
# Location: /media/isodev/.../Nova/Embedding

from sentence_transformers import SentenceTransformer
import numpy as np

# Load Gemma small embedding model (fits Pi 5 RAM)
print("⏳ Loading EmbeddingGemma model...")
model = SentenceTransformer("google/embeddinggemma-300m")  # Tiny variant

# Your sentence groups
sentences_list = [
    ["The chef prepared a delicious meal for the guests.",
     "A tasty dinner was cooked by the chef for the visitors."],

    ["She is an expert in machine learning.",
     "He has a deep interest in artificial intelligence."],

    ["The weather in Tokyo is sunny today.",
     "I need to buy groceries for the week."]
]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Encode and calculate similarities
for sentences in sentences_list:
    print("\n🙋‍♂️ Sentence group:")
    for s in sentences:
        print(" -", s)
    
    embeddings = model.encode(sentences)
    sim_score = cosine_similarity(embeddings[0], embeddings[1])
    print("`-> 🤖 Similarity score:", sim_score)
