from sentence_transformers import SentenceTransformer

from step1_load_corpus import load_corpus
from step2c_chunk_all import chunk_text

model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = []
for doc in load_corpus():
    for piece in chunk_text(doc["text"], 800, 100):
        chunks.append({"source_id": doc["id"], "text": piece})

vectors = model.encode([c["text"] for c in chunks], normalize_embeddings=True)

print("chunks:", len(chunks))
print("vectors shape:", vectors.shape)