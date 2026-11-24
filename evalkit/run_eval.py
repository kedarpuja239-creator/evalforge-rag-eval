"""Run the golden set through the assistant and write results/latest.json.

    python -m evalkit.run_eval               retrieval only: recall@k, hit@k, MRR (local, free, no key)
    python -m evalkit.run_eval --answers     also generate answers and grade them with an LLM judge
                                             (faithfulness, answer_relevancy; needs a working LLM key)
"""
from __future__ import annotations

import argparse
import json
import sys

from . import config
from .index import Retriever
from .metrics import hit_at_k, mean, recall_at_k, reciprocal_rank


def load_golden() -> list[dict]:
    return [json.loads(line) for line in config.GOLDEN_PATH.read_text("utf-8").splitlines() if line.strip()]


def evaluate_retrieval(retriever: Retriever, golden: list[dict], k: int) -> tuple[dict[str, float], list[dict]]:
    rr, rec, hit, rows = [], [], [], []
    for row in golden:
        results = retriever.search(row["question"], k)
        ids = [chunk.source_id for _, chunk in results]
        rows.append({"id": row["id"], "retrieved_ids": ids})
        relevant = set(row.get("reference_doc_ids", []))
        if relevant:  # unanswerable questions have no retrieval ground truth
            rr.append(reciprocal_rank(ids, relevant))
            rec.append(recall_at_k(ids, relevant, k))
            hit.append(hit_at_k(ids, relevant, k))
    return {f"recall@{k}": mean(rec), f"hit@{k}": mean(hit), "mrr": mean(rr)}, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true", help="also generate answers and run the LLM judge")
    ap.add_argument("--limit", type=int, help="only the first N golden questions (cheap trial run)")
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "latest.json"))
    args = ap.parse_args()

    golden = load_golden()[: args.limit]
    retriever = Retriever.load()
    scores, rows = evaluate_retrieval(retriever, golden, config.TOP_K)
    samples = [{**g, **r} for g, r in zip(golden, rows)]

    if args.answers:
        from . import judge
        from .llm import friendly_error
        from .rag import answer

        try:
            faithful, relevant = [], []
            for s in samples:
                ctxs = [chunk for _, chunk in retriever.search(s["question"], config.TOP_K)]
                s["contexts"] = [c.text for c in ctxs]
                s["response"] = answer(s["question"], ctxs)
                g = judge.grade(s["question"], "\n---\n".join(s["contexts"]), s["response"])
                s["judge"] = g
                faithful.append(g["faithful"])
                relevant.append(g["relevant"])
            scores["faithfulness"] = mean(faithful)
            scores["answer_relevancy"] = mean(relevant)
        except Exception as e:
            print("LLM step failed:", friendly_error(e))
            return 2

    out = {
        "config": {"strategy": config.CHUNK_STRATEGY, "chunk_size": config.CHUNK_SIZE,
                   "overlap": config.CHUNK_OVERLAP, "top_k": config.TOP_K, "embed_model": config.EMBED_MODEL,
                   "gen_model": config.GEN_MODEL if args.answers else None,
                   "judge_model": config.JUDGE_MODEL if args.answers else None, "n": len(samples)},
        "scores": scores,
        "samples": samples,
    }
    config.RESULTS_DIR.mkdir(exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(scores, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
