"""Pool test on PACE: evidence sets, expert pool and yes/no matching, with a greedy judge.
Writes <out>/pool_test/{summary,rows,sets,verdicts}.json; replies are cached in one file."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from georc.extras import RELEASED, CacheOnly
from georc.judge import CachedJudge
from georc.models import JUDGE_LARGE
from georc.pool import run, table
from georc.vllm_backend import VLLMJudge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sets", nargs="+", default=list(RELEASED))
    parser.add_argument("--cue-template", default="same_cue.txt")
    parser.add_argument("--both-orders", action="store_true")
    parser.add_argument("--name", default="pool_test", help="output folder under --out")
    parser.add_argument(
        "--large-matcher",
        action="store_true",
        help="match with the 30B judge; splitting is read from the 4B judge's cache",
    )
    args = parser.parse_args()
    cache = args.out / "judge_cache"
    if args.large_matcher:
        judge = CachedJudge(CacheOnly(), cache / "pool_greedy.jsonl")
        matcher = CachedJudge(
            VLLMJudge(seed=0, greedy=True, spec=JUDGE_LARGE, max_model_len=8192),
            cache / "pool_large.jsonl",
        )
    else:
        judge = matcher = CachedJudge(VLLMJudge(seed=0, greedy=True), cache / "pool_greedy.jsonl")
    result = run(
        args.data, args.out, judge, args.sets, args.cue_template, args.both_orders, matcher
    )
    print(table(result["summary"]), flush=True)
    folder = args.out / args.name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(result["summary"], indent=1))
    (folder / "rows.json").write_text(json.dumps(result["rows"]))
    sets = {name: [asdict(s) for s in found] for name, found in result["sets"].items()}
    (folder / "sets.json").write_text(json.dumps(sets))
    verdicts = {
        kind: [[a, b, v] for (a, b), v in found.items()]
        for kind, found in result["verdicts"].items()
    }
    (folder / "verdicts.json").write_text(json.dumps(verdicts))


if __name__ == "__main__":
    main()
