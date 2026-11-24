"""Before/after proof for retrieval: score several chunking setups on the golden set, no API key needed.

    python -m evalkit.compare            writes results/chunking_comparison.json and prints a table

Change ONE thing at a time (strategy or size) and read the difference: that is the experiment.
"""
from __future__ import annotations

import json

from . import config
from .index import Retriever, build_chunks
from .run_eval import evaluate_retrieval, load_golden

# (strategy, size, overlap). "token" counts tokens, the others count characters.
SETUPS = [
    ("fixed", 800, 100),
    ("fixed", 400, 50),
    ("recursive", 400, 50),
    ("semantic", 400, 50),
    ("token", 100, 10),
]


def main() -> None:
    golden = load_golden()
    k = config.TOP_K
    table = []
    for strategy, size, overlap in SETUPS:
        try:
            chunks = build_chunks(strategy, size, overlap)
            scores, _ = evaluate_retrieval(Retriever.from_chunks(chunks), golden, k)
            table.append({"strategy": strategy, "size": size, "overlap": overlap, "chunks": len(chunks), **scores})
        except Exception as e:  # for example tiktoken cannot download its vocabulary offline
            table.append({"strategy": strategy, "size": size, "overlap": overlap, "error": str(e)[:120]})

    print(f"{'strategy':<10} {'size':>5} {'ovl':>4} {'chunks':>6} {f'recall@{k}':>9} {'mrr':>6}")
    for r in table:
        if "error" in r:
            print(f"{r['strategy']:<10} {r['size']:>5} {r['overlap']:>4}   skipped: {r['error']}")
        else:
            print(f"{r['strategy']:<10} {r['size']:>5} {r['overlap']:>4} {r['chunks']:>6} {r[f'recall@{k}']:>9.3f} {r['mrr']:>6.3f}")
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "chunking_comparison.json").write_text(json.dumps(table, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
