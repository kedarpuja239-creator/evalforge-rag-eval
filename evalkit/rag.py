"""The assistant: retrieve the top-k chunks, then ask an LLM to answer ONLY from them.

    python -m evalkit.rag "How do I check my balance?"     (needs a working LLM key)
"""
from __future__ import annotations

import sys

from . import config
from .index import Chunk, Retriever
from .llm import client

SYSTEM = (
    "You are a telecom customer-support assistant. Answer ONLY from the provided context. "
    "If the context does not contain the answer, say you don't know. Be concise."
)


def answer(question: str, contexts: list[Chunk]) -> str:
    ctx = "\n\n".join(f"[{c.source_id}]\n{c.text}" for c in contexts)
    resp = client().chat.completions.create(
        model=config.GEN_MODEL,
        temperature=0,  # as repeatable as possible, which a regression gate needs
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content or ""


def ask(question: str, retriever: Retriever | None = None) -> tuple[str, list[Chunk]]:
    retriever = retriever or Retriever.load()
    contexts = [chunk for _, chunk in retriever.search(question)]
    return answer(question, contexts), contexts


if __name__ == "__main__":
    reply, ctxs = ask(" ".join(sys.argv[1:]))
    print(reply, "\n\nsources:", [c.source_id for c in ctxs])
