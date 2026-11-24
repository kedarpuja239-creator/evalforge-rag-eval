"""Label the judge-calibration rows in the terminal. Saves after every answer, so you can stop and resume.

    python -m evalkit.label            # labels data/judge_labels.csv
    python -m evalkit.label file.csv   # labels another file (used for testing)

Give 1 only if ALL three are true:
  - the answer is supported by the context and adds nothing the context does not say
  - it correctly answers the question
  - if the context does not contain the answer, the answer says "I don't know" or similar
Otherwise give 0. Decide from the context and the answer only, not from what you know about Vi.
"""
from __future__ import annotations

import csv
import sys
import textwrap

from . import config


def _load(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def _save(path, fields, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(path=None) -> None:
    path = path or config.JUDGE_LABELS_PATH
    fields, rows = _load(path)
    todo = [i for i, r in enumerate(rows) if r["human_label"].strip() not in ("0", "1")]
    print(f"{len(rows) - len(todo)} labelled, {len(todo)} to go.")
    print("Type 1 (correct AND grounded), 0 (not), s (skip), or q (save and quit).")
    for n, i in enumerate(todo, start=1):
        row = rows[i]
        print("\n" + "=" * 78)
        print(f"[{n}/{len(todo)}] {row['id']}")
        print("QUESTION:", row["question"])
        print("CONTEXT:")
        print(textwrap.indent(textwrap.fill(row["context"], 90, replace_whitespace=False), "   "))
        print("ANSWER:", row["answer"])
        while True:
            try:
                choice = input("label (1/0/s/q) > ").strip().lower()
            except EOFError:
                choice = "q"
            if choice in ("0", "1"):
                row["human_label"] = choice
                _save(path, fields, rows)
                break
            if choice == "s":
                break
            if choice == "q":
                _save(path, fields, rows)
                done = sum(r["human_label"].strip() in ("0", "1") for r in rows)
                print(f"saved. {done}/{len(rows)} labelled.")
                return
            print("please type 1, 0, s or q")
    done = sum(r["human_label"].strip() in ("0", "1") for r in rows)
    print(f"\nfinished: {done}/{len(rows)} labelled. Next: python -m evalkit.judge calibrate")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
