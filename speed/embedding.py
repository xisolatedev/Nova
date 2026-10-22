
import argparse
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--top-k", type=int, default=3)
args = parser.parse_args()

if args.threads < 1 or args.top_k < 1:
    parser.error("Threads and top-k must be positive")

torch.set_num_threads(args.threads)

checkpoint = "Qwen/Qwen3-Embedding-0.6B"


print(f"Loading {checkpoint} on {args.device}...")

start = time.perf_counter()

model = SentenceTransformer(
    checkpoint,
    device=args.device,
    model_kwargs={"torch_dtype": torch.float32}
)

load_time = time.perf_counter() - start

documents = [
    "Gravity is the force that attracts objects with mass.",
    "Python is a programming language used for AI development.",
    "Neural networks learn patterns from training data.",
    "The Earth orbits around the Sun.",
    "Graphics cards accelerate parallel computations.",
    "Transformers use attention mechanisms to process information.",
    "Mixture of Experts models activate selected neural network experts.",
    "Machine learning models can recognize objects in images.",
    "Linux is an operating system used on computers and servers.",
    "Embeddings represent information as numerical vectors."
]

print("Creating document embeddings...")

start = time.perf_counter()

with torch.inference_mode():
    document_vectors = model.encode_document(
        documents,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )

index_time = time.perf_counter() - start

print(f"Loaded in {load_time:.2f}s")
print(f"Indexed {len(documents)} documents in {index_time:.3f}s")
print(f"Embedding dimensions: {document_vectors.shape[1]}")

if args.verbose:
    parameters = sum(p.numel() for p in model.parameters())

    print("\n[DEBUG] Model")
    print(f"  Parameters: {parameters:,}")
    print(f"  Device: {args.device}")
    print(f"  CPU threads: {args.threads}")
    print(f"  Model load time: {load_time:.3f}s")
    print(f"  Indexing speed: {len(documents) / index_time:.2f} docs/sec")

while True:
    query = input("\nSearch: ").strip()

    if query.lower() in ("/quit", "/exit"):
        break

    if not query:
        continue

    start = time.perf_counter()

    with torch.inference_mode():
        query_vector = model.encode_query(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

    embedding_time = time.perf_counter() - start

    # Both vectors are normalized, so dot product
    # is equivalent to cosine similarity.
    search_start = time.perf_counter()

    scores = document_vectors @ query_vector

    ranking = np.argsort(scores)[::-1][:args.top_k]

    search_time = time.perf_counter() - search_start

    print("\nResults:")

    for rank, index in enumerate(ranking, start=1):
        print(f"{rank}. [{scores[index]:.4f}] {documents[index]}")

    total_time = time.perf_counter() - start

    print(f"\n[Speed] {total_time * 1000:.2f} ms/query")

    if args.verbose:
        print("\n[DEBUG] Performance")
        print(f"  Query embedding: {embedding_time * 1000:.2f} ms")
        print(f"  Vector search: {search_time * 1000:.3f} ms")
        print(f"  Total query time: {total_time * 1000:.2f} ms")
        print(f"  Query throughput: {1 / total_time:.2f} queries/sec")
        print(f"  Documents searched: {len(documents)}")
        print(f"  Vector dimensions: {query_vector.shape[0]}")

        if args.device == "cuda":
            print(
                f"  GPU memory: "
                f"{torch.cuda.memory_allocated() / 1024**2:.1f} MiB"
            )
