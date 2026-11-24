from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "How do I check my balance?",
    "See my remaining credit",
    "How do I build a rocket ship?",
]
vectors = model.encode(texts, normalize_embeddings=True)

print("shape:", vectors.shape)
print("0 vs 1:", float(vectors[0] @ vectors[1]))
print("0 vs 2:", float(vectors[0] @ vectors[2]))