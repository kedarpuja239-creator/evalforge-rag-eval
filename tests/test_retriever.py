import numpy as np

from evalkit.index import Chunk, Retriever
from evalkit.run_eval import evaluate_retrieval

KEYWORDS = ["balance", "bill", "roaming"]


def fake_embed(texts):
    """One dimension per keyword, so no model download is needed in unit tests."""
    vecs = np.array([[t.lower().count(k) for k in KEYWORDS] for t in texts], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1, norms)


CHUNKS = [
    Chunk("balance-page", "check your balance with *199#"),
    Chunk("bill-page", "pay your postpaid bill online"),
    Chunk("roaming-page", "activate international roaming pack"),
]


def test_search_returns_most_similar_chunk_first():
    r = Retriever.from_chunks(CHUNKS, fake_embed)
    results = r.search("how do I pay my bill", k=2)
    assert [c.source_id for _, c in results][0] == "bill-page"
    assert len(results) == 2
    assert results[0][0] >= results[1][0]


def test_evaluate_retrieval_scores():
    r = Retriever.from_chunks(CHUNKS, fake_embed)
    golden = [
        {"id": "g1", "question": "check balance", "reference_doc_ids": ["balance-page"]},
        {"id": "g2", "question": "roaming pack", "reference_doc_ids": ["roaming-page"]},
        {"id": "g3", "question": "something unrelated", "reference_doc_ids": []},  # unanswerable: skipped
    ]
    scores, rows = evaluate_retrieval(r, golden, 1)
    assert scores["recall@1"] == 1.0 and scores["mrr"] == 1.0
    assert len(rows) == 3
