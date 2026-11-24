"""CI gate: exit 1 if any score is below its floor or has dropped too far from the committed baseline.

    python -m evalkit.gate [results/latest.json] [--thresholds thresholds.json] [--update-baseline]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import config

THRESHOLDS = config.ROOT / "thresholds.json"  # absolute floors
BASELINE = config.RESULTS_DIR / "baseline.json"  # last accepted scores
MAX_DROP = 0.03  # allowed absolute drop vs baseline, per metric


def check(scores: dict[str, float], floors: dict[str, float], baseline: dict[str, float]) -> list[str]:
    failures = []
    for metric, floor in floors.items():
        if metric not in scores:
            failures.append(f"{metric}: missing from results")
        elif scores[metric] < floor:
            failures.append(f"{metric}: {scores[metric]:.3f} < floor {floor:.3f}")
    for metric, base in baseline.items():
        if metric in scores and scores[metric] < base - MAX_DROP:
            failures.append(f"{metric}: {scores[metric]:.3f} regressed from baseline {base:.3f} (max drop {MAX_DROP})")
    return failures


def main() -> int:
    argv = sys.argv[1:]
    thresholds = THRESHOLDS
    if "--thresholds" in argv:
        i = argv.index("--thresholds")
        thresholds = Path(argv[i + 1])
        del argv[i : i + 2]
    args = [a for a in argv if not a.startswith("--")]
    path = Path(args[0]) if args else config.RESULTS_DIR / "latest.json"
    scores = json.loads(path.read_text("utf-8"))["scores"]
    if "--update-baseline" in argv:
        BASELINE.parent.mkdir(exist_ok=True)
        BASELINE.write_text(json.dumps(scores, indent=2), encoding="utf-8")
        print("baseline updated")
        return 0
    floors = json.loads(thresholds.read_text("utf-8"))
    baseline = json.loads(BASELINE.read_text("utf-8")) if BASELINE.exists() else {}
    failures = check(scores, floors, baseline)
    for name, value in scores.items():
        print(f"{name:>18}: {value:.3f}")
    if failures:
        print("\nEVAL GATE FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("\nEVAL GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
