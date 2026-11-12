"""Score released (--released) or generated (--dirs) chain sets with the vLLM judge (PACE)."""

import argparse
import json
from pathlib import Path

from georc.data import load_pairs, load_rounds
from georc.evaluate import directory_chain_set, evaluate, released_chain_set, write_rows
from georc.generate import pilot_rounds, restrict_pairs
from georc.judge import PROMPT_VARIANTS, CachedJudge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--released", nargs="*", default=[])
    parser.add_argument("--dirs", nargs="*", default=[], help="name=folder pairs")
    parser.add_argument("--variant", choices=list(PROMPT_VARIANTS), default="paper")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="first 20 challenges only")
    args = parser.parse_args()

    from georc.vllm_backend import VLLMJudge  # vllm exists only on PACE

    rounds = load_rounds(args.data)
    pairs = load_pairs(args.data, rounds)
    if args.pilot:
        rounds = pilot_rounds(rounds)
        pairs = restrict_pairs(pairs, rounds)
    # One cache per judge setting (not per run), so pilot and full runs share replies.
    judge_tag = f"{args.variant}_s{args.seed}{'_greedy' if args.greedy else ''}"
    tag = judge_tag + ("_pilot" if args.pilot else "")
    judge = CachedJudge(
        VLLMJudge(seed=args.seed, greedy=args.greedy),
        args.out / "judge_cache" / f"{judge_tag}.jsonl",
    )
    chain_sets = [(name, released_chain_set(args.data, rounds, name)) for name in args.released]
    for spec in args.dirs:
        name, folder = spec.split("=", 1)
        chain_sets.append((name, directory_chain_set(Path(folder), rounds)))

    for name, chains in chain_sets:
        pair_rows, round_rows, summary = evaluate(chains, pairs, rounds, judge, args.variant)
        folder = args.out / "scores" / f"{name.replace('/', '_')}__{tag}"
        write_rows(folder / "pairs.csv", pair_rows)
        write_rows(folder / "rounds.csv", round_rows)
        (folder / "summary.json").write_text(json.dumps({"name": name, **summary}, indent=2))
        print(f"{name:24s} {tag}: " + ", ".join(f"{k}={v:.2f}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
