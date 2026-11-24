from step1_load_corpus import load_corpus

docs = load_corpus()
text = docs[0]["text"]

print("Total length:", len(text))
print("First 100 chars:", repr(text[0:100]))
print("Chars 50 to 150:", repr(text[50:150]))
print("Last 40 chars:", repr(text[-40:]))