"""Chunking strategies. Every strategy has the same signature, (text, size, overlap) -> list[str],
so they are interchangeable. Pick one with EVALKIT_CHUNK_STRATEGY.

    fixed      sliding character window (what you wrote in step2c)
    recursive  cut at paragraph, then line, then sentence, then word boundaries
    semantic   same splitter with a finer separator list (still separator-based, not embedding-based)
    token      sliding window over TOKENS (cl100k_base); size/overlap count tokens, not characters
    agentic    an LLM chooses where to cut (slow, costs API calls); falls back to recursive on any failure

Ideas adapted from the RAG Fitness Agent tutorial by Nisarg Kadam (MIT licence).
"""
from __future__ import annotations

import json
from typing import Callable

Chunker = Callable[[str, int, int], "list[str]"]

RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
SEMANTIC_SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
AGENTIC_MAX_INPUT_CHARS = 20_000


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
        chunk_size=size, chunk_overlap=overlap, separators=SEMANTIC_SEPARATORS,
        length_function=len, keep_separator=True,
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
    """The LLM picks line numbers (not character offsets: LLMs cannot count characters reliably)."""
    if len(text) <= size:
        return [text] if text.strip() else []
    if len(text) > AGENTIC_MAX_INPUT_CHARS:
        return recursive(text, size, overlap)
    try:
        from . import config
        from .llm import client

        lines = text.splitlines()
        numbered = "\n".join(f"{i}: {line}" for i, line in enumerate(lines))
        resp = client().chat.completions.create(
            model=config.GEN_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": _AGENTIC_PROMPT.format(size=size, numbered=numbered)}],
        )
        raw = json.loads(resp.choices[0].message.content)["starts"]
        starts = sorted({int(n) for n in raw if 0 < int(n) < len(lines)})
        bounds = [0, *starts, len(lines)]
        groups = ["\n".join(lines[a:b]).strip() for a, b in zip(bounds, bounds[1:])]
        chunks: list[str] = []
        for prev, group in zip([""] + groups, groups):
            if not group:
                continue
            piece = (prev[-overlap:] + "\n" + group).strip() if overlap and prev else group
            chunks += recursive(piece, size, overlap) if len(piece) > size * 1.5 else [piece]
        return chunks or recursive(text, size, overlap)
    except Exception as e:
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
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown chunk strategy {strategy!r}; choose from {sorted(STRATEGIES)}")
    return STRATEGIES[strategy](text, size, overlap)
