"""Extra judge passes for alternative metrics (PACE). Only the reply caches matter: one file per
(pass, chain set) under <out>/judge_cache/extras/, so jobs run in parallel and resume."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from georc.data import load_rounds
from georc.extras import (
    EXPERTS,
    GROUNDED_MAX_TOKENS,
    PASSES,
    SETS,
    cache_path,
    call_counts,
    describe,
    run_pass,
)
from georc.judge import CachedJudge, Judge


class RecordingJudge:
    """Dry runs: keeps every prompt and answers each with an empty reply."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompts: Sequence[str]) -> list[str]:
        self.prompts += prompts
        return [""] * len(prompts)


def load_judge(pass_name: str, data: Path) -> Judge:
    from georc.vllm_backend import VLLMJudge, VLLMVisionJudge  # vllm exists only on PACE

    if pass_name == "grounded":
        return VLLMVisionJudge({r.key: r.image for r in load_rounds(data)}, GROUNDED_MAX_TOKENS)
    return VLLMJudge(seed=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pass", dest="pass_name", choices=PASSES, required=True)
    parser.add_argument("--sets", nargs="+", choices=SETS, help="default: all the pass takes")
    parser.add_argument("--dry-run", action="store_true", help="count judge calls, no GPU")
    args = parser.parse_args()
    sets = args.sets or ([EXPERTS] if args.pass_name == "experts" else list(SETS))
    if args.pass_name == "experts" and sets != [EXPERTS]:
        parser.error("--pass experts takes only --sets experts")

    judge = None
    for name in sets:
        path = cache_path(args.out, args.pass_name, name)
        if args.dry_run:
            recorder = RecordingJudge()
            run_pass(args.pass_name, name, args.data, args.out, recorder)
            counts = call_counts(recorder.prompts, path)
            print(f"{args.pass_name} {name}: " + " ".join(f"{k}={v}" for k, v in counts.items()))
            continue
        if judge is None:
            judge = load_judge(args.pass_name, args.data)
        # Grounded requests each carry an image, so they go out (and are saved) in smaller chunks.
        chunk = 128 if args.pass_name == "grounded" else 2000
        result = run_pass(
            args.pass_name, name, args.data, args.out, CachedJudge(judge, path, chunk)
        )
        print(describe(args.pass_name, name, result), flush=True)


if __name__ == "__main__":
    main()
