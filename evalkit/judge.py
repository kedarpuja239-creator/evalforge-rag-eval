"""LLM-as-judge, calibrated against your own labels.

    python -m evalkit.judge sample      writes data/judge_labels.csv: 50 rows with an empty human_label column.
                                        Fill it with 1 (answer is correct AND grounded) or 0 WITHOUT looking at
                                        what the judge says. Needs results/latest.json from `run_eval --answers`.
    python -m evalkit.judge calibrate   runs the judge on your labelled rows, writes results/judge_calibration.json
                                        (agreement %, Cohen's kappa, confusion matrix)
"""
from __future__ import annotations

import csv
import json
import random
import sys

from . import config
from .llm import client

RUBRIC = """You grade a customer-support answer against the retrieved context.
Return JSON {{"reason": str, "faithful": 1 or 0, "relevant": 1 or 0, "verdict": 1 or 0}}.
faithful=1 only if every claim in the answer is supported by the context (an answer that says it does not know is faithful).
relevant=1 only if the answer addresses the question that was asked.
verdict=1 only if the answer is supported by the context, correctly answers the question, and says
"I don't know" when the context lacks the answer. Otherwise 0.

Question: {question}
Context: {context}
Answer: {answer}"""


def grade(question: str, context: str, answer: str) -> dict:
    resp = client().chat.completions.create(
        model=config.JUDGE_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": RUBRIC.format(question=question, context=context, answer=answer)}],
    )
    data = json.loads(resp.choices[0].message.content)
    return {k: int(data[k]) for k in ("faithful", "relevant", "verdict")} | {"reason": data.get("reason", "")}


def sample(n: int = 50) -> None:
    data = json.loads((config.RESULTS_DIR / "latest.json").read_text("utf-8"))["samples"]
    data = [s for s in data if "response" in s]
    if not data:
        sys.exit("latest.json has no answers: run `python -m evalkit.run_eval --answers` first")
    random.Random(42).shuffle(data)
    with open(config.JUDGE_LABELS_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "context", "answer", "human_label"])
        for s in data[:n]:
            w.writerow([s["id"], s["question"], "\n---\n".join(s["contexts"]), s["response"], ""])
    print(f"wrote {min(n, len(data))} rows to {config.JUDGE_LABELS_PATH}; fill human_label (1/0)")


def cohen_kappa(a: list[int], b: list[int]) -> float:
    """Agreement corrected for chance: 1 = perfect, 0 = no better than guessing."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in (0, 1))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def calibrate() -> dict:
    with open(config.JUDGE_LABELS_PATH, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["human_label"].strip() in ("0", "1")]
    if not rows:
        sys.exit("no labelled rows: fill the human_label column with 1 or 0")
    if len(rows) < 50:
        print(f"warning: only {len(rows)} labelled rows (target 50)")
    human = [int(r["human_label"]) for r in rows]
    judged = [grade(r["question"], r["context"], r["answer"])["verdict"] for r in rows]
    agree = sum(h == j for h, j in zip(human, judged)) / len(rows)
    report = {
        "n": len(rows),
        "agreement_pct": round(100 * agree, 1),
        "cohen_kappa": round(cohen_kappa(human, judged), 3),
        "confusion": {
            "tp": sum(h == j == 1 for h, j in zip(human, judged)),
            "tn": sum(h == j == 0 for h, j in zip(human, judged)),
            "fp": sum(h == 0 and j == 1 for h, j in zip(human, judged)),
            "fn": sum(h == 1 and j == 0 for h, j in zip(human, judged)),
        },
    }
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "judge_calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    {"sample": sample, "calibrate": calibrate}[sys.argv[1]]()
