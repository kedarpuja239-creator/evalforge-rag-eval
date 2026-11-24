"""Chunking strategies for the RAG index. Pick one with EVALFORGE_CHUNK_STRATEGY.

    fixed      sliding character window (the original behaviour, default so existing baselines still hold)
    recursive  LangChain recursive splitter: paragraphs -> lines -> sentences -> words
    semantic   recursive splitter with a finer separator list that keeps the separators
    token      token-based windows (cl100k_base); size/overlap are TOKENS, not characters
    agentic    an LLM picks the break points (slow and costs API calls); falls back to recursive

All strategies share one signature, (text, size, overlap) -> list[str], so they are interchangeable in
build_index(). Ground truth in the golden set is document-level, so changing strategy never invalidates it.
"""
from __future__ import annotations

import json
import re
from typing import Callable

Chunker = Callable[[str, int, int], list[str]]

SEMANTIC_SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
AGENTIC_MAX_INPUT_CHARS = 20_000  # longer docs skip the LLM and use the recursive fallback


def fixed(text: str, size: int, overlap: int) -> list[str]:
    step = max(size - overlap, 1)
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


def recursive(text: str, size: int, overlap: int) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=RECURSIVE_SEPARATORS, length_function=len
    )
    return [c for c in splitter.split_text(text) if c.strip()]


def semantic(text: str, size: int, overlap: int) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=SEMANTIC_SEPARATORS,
        length_function=len,
        keep_separator=True,
    )
    return [c for c in splitter.split_text(text) if c.strip()]


def token(text: str, size: int, overlap: int) -> list[str]:
    from langchain_text_splitters import TokenTextSplitter

    splitter = TokenTextSplitter(chunk_size=size, chunk_overlap=overlap, encoding_name="cl100k_base")
    return [c for c in splitter.split_text(text) if c.strip()]


_AGENTIC_PROMPT = """You split a document into semantically coherent chunks of about {size} characters each.
The document is shown as numbered lines. Choose the line numbers where a NEW chunk should start.
Rules: never split mid-topic; keep a question with its answer; chunks may vary by about 20%.
Return JSON only: {{"starts": [<line numbers, ascending, excluding the first line>]}}

{numbered}"""


def agentic(text: str, size: int, overlap: int) -> list[str]:
    """LLM picks line numbers where chunks start. Line numbers (not character offsets) because LLMs
    are unreliable at counting characters. Any failure falls back to the recursive splitter."""
    if len(text) <= size:
        return [text] if text.strip() else []
    if len(text) > AGENTIC_MAX_INPUT_CHARS:
        return recursive(text, size, overlap)
    try:
        from . import config
        from .rag import client  # imported lazily: rag imports this module

        lines = text.splitlines()
        numbered = "\n".join(f"{i}: {line}" for i, line in enumerate(lines))
        resp = client().chat.completions.create(
            model=config.GEN_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": _AGENTIC_PROMPT.format(size=size, numbered=numbered)}],
        )
        starts = sorted({int(n) for n in json.loads(resp.choices[0].message.content)["starts"] if 0 < int(n) < len(lines)})
        bounds = [0, *starts, len(lines)]
        groups = ["\n".join(lines[a:b]).strip() for a, b in zip(bounds, bounds[1:])]
        chunks: list[str] = []
        for prev, group in zip([""] + groups, groups):
            if not group:
                continue
            piece = (prev[-overlap:] + "\n" + group).strip() if overlap and prev else group
            # a group the LLM left far too big is re-split mechanically
            chunks += recursive(piece, size, overlap) if len(piece) > size * 1.5 else [piece]
        return chunks or recursive(text, size, overlap)
    except Exception as e:  # network, bad JSON, wrong keys ...
        print(f"agentic chunking failed, using recursive fallback: {e}")
        return recursive(text, size, overlap)


STRATEGIES: dict[str, Chunker] = {
    "fixed": fixed,
    "recursive": recursive,
    "semantic": semantic,
    "token": token,
    "agentic": agentic,
}


def chunk(text: str, size: int, overlap: int, strategy: str = "fixed") -> list[str]:
    try:
        chunker = STRATEGIES[strategy]
    except KeyError:
        raise ValueError(f"unknown chunk strategy {strategy!r}; choose from {sorted(STRATEGIES)}") from None
    return chunker(text, size, overlap)


def compare(text: str, size: int = 800, overlap: int = 100) -> dict[str, dict]:
    """Chunk count / average length / first chunk per local strategy, to eyeball differences.
    Agentic is excluded because it costs API calls; call chunk(..., 'agentic') explicitly."""
    out = {}
    for name in ("fixed", "recursive", "semantic", "token"):
        chunks = chunk(text, size, overlap, name)
        out[name] = {
            "chunks": len(chunks),
            "avg_len": sum(map(len, chunks)) // len(chunks) if chunks else 0,
            "first": re.sub(r"\s+", " ", chunks[0])[:150] if chunks else "",
        }
    return out
