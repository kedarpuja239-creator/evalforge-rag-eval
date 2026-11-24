import pytest

from evalkit import chunking

TEXT = "\n\n".join(f"Question {i}? " + "Answer sentence. " * 15 for i in range(12))


def test_fixed_window_lengths_and_overlap():
    out = chunking.fixed("a" * 1000, 400, 100)
    assert [len(c) for c in out] == [400, 400, 400, 100]
    assert chunking.fixed("   ", 10, 2) == []


def test_fixed_chunks_overlap_by_the_overlap_amount():
    text = "".join(chr(97 + i % 26) for i in range(300))
    a, b = chunking.fixed(text, 100, 20)[:2]
    assert a[-20:] == b[:20]


def test_short_text_stays_one_chunk():
    assert chunking.fixed("short page", 800, 100) == ["short page"]


@pytest.mark.parametrize("strategy", ["recursive", "semantic"])
def test_splitters_respect_size_and_keep_all_content(strategy):
    out = chunking.chunk(TEXT, 300, 20, strategy)
    assert len(out) > 1 and max(map(len, out)) <= 300
    joined = "\n".join(out)
    assert all(f"Question {i}?" in joined for i in range(12))


def test_agentic_short_text_needs_no_llm():
    assert chunking.agentic("short text", 800, 100) == ["short text"]


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown chunk strategy"):
        chunking.chunk("x", 10, 1, "nope")
