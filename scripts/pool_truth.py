"""Check every evidence set's location against the round's true location with the 30B judge (PACE).
Reads <out>/<name>/sets.json from a pool test and writes <out>/<name>/truth.json."""

import argparse
import json
from pathlib import Path

from georc.data import load_coordinates, load_rounds
from georc.judge import CachedJudge
from georc.models import JUDGE_LARGE
from georc.pool import EvidenceSet, ask_truth
from georc.vllm_backend import VLLMJudge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default="pool_large_strict")
    args = parser.parse_args()
    folder = args.out / args.name
    found = json.loads((folder / "sets.json").read_text())
    sets = [EvidenceSet(**s) for group in found.values() for s in group]
    coordinates = load_coordinates(args.data)
    places = {r.key: (*coordinates[r.key], r.country) for r in load_rounds(args.data)}
    judge = CachedJudge(
        VLLMJudge(seed=0, greedy=True, spec=JUDGE_LARGE, max_model_len=8192),
        args.out / "judge_cache" / "truth_large.jsonl",
    )
    verdicts = ask_truth(sets, places, judge)
    rows = [[key, guess, verdict] for (key, guess), verdict in verdicts.items()]
    (folder / "truth.json").write_text(json.dumps(rows))
    known = [v for v in verdicts.values() if v is not None]
    print(
        f"guesses={len(rows)} correct={sum(known)} unreadable={len(rows) - len(known)}", flush=True
    )


if __name__ == "__main__":
    main()
