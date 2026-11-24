from pathlib import Path

CORPUS_DIR = Path("data/corpus")

def load_all():
    docs = []
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        first_line, separator, body = text.partition("\n")
        source = first_line.replace("SOURCE:", "").strip()
        docs.append({"id": path.stem, "source": source, "text": body.strip()})
    return docs

docs = load_all()
print("Documents loaded:", len(docs))