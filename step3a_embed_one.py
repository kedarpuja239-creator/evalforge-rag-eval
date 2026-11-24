from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
vector = model.encode("How do I check my balance?")

print("type:", type(vector))
print("length:", len(vector))
print("first 5 numbers:", vector[:5])