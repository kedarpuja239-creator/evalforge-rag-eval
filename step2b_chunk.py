from step1_load_corpus import load_corpus
SIZE=100
OVERLAP=20

docs=load_corpus()
text=docs[0]["text"]
step=SIZE-OVERLAP
chunks=[]

for start in range(0, len(text), step):
    chunks.append(text[start:start+SIZE])
    print("Step:",step)
    print("Number of chunks:",len(chunks))
    for i, chunk in enumerate(chunks):
        print(i,len(chunk),repr(chunk))
        