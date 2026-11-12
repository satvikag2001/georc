"""Generate chains for one condition and one or more seeds (PACE); existing chains are kept."""

import argparse
from pathlib import Path

from georc.conditions import CONDITIONS
from georc.data import load_rounds
from georc.generate import generate, pending_rounds, pilot_rounds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--pilot", action="store_true", help="first 20 challenges only")
    parser.add_argument("--dry-run", action="store_true", help="count pending chains, no GPU")
    args = parser.parse_args()

    condition = CONDITIONS[args.condition]
    rounds = load_rounds(args.data)
    if args.pilot:
        rounds = pilot_rounds(rounds)
    for seed in args.seeds:
        todo = pending_rounds(rounds, args.out, condition.id, seed)
        print(f"{condition.id} seed{seed}: {len(todo)} of {len(rounds)} chains to generate")
        if not args.dry_run:
            generate(condition, rounds, args.out, seed)


if __name__ == "__main__":
    main()
