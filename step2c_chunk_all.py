from step1_load_corpus import load_corpus


def chunk_text(text, size, overlap):
    step = max(size - overlap, 1)
    chunks = []
    for start in range(0, len(text), step):
        piece = text[start:start + size]
        if piece.strip():
            chunks.append(piece)
    return chunks


if __name__ == "__main__":
    docs = load_corpus()
    all_chunks = []
    for doc in docs:
        for piece in chunk_text(doc["text"], 800, 100):
            all_chunks.append({"source_id": doc["id"], "text": piece})

    print("Documents:", len(docs))
    print("Chunks:", len(all_chunks))
    print("Longest chunk:", max(len(c["text"]) for c in all_chunks))
    print("Shortest chunk:", min(len(c["text"]) for c in all_chunks))