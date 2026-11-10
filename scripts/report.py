"""Turn score summaries pulled from PACE into result tables and gate verdicts.

bash pace/pull.sh && report.py --validation runs/georc/scores   # gate G1 on released chains
"""

import argparse
from pathlib import Path

from georc.evaluate import load_summaries, write_rows
from georc.gates import OPEN_MODELS, check_country, check_f1, validation_rows


def report_validation(scores: Path, tag: str, noise_tag: str, out: Path) -> bool:
    summaries = load_summaries(scores, tag)
    rows = validation_rows(summaries)
    write_rows(out, rows)
    print(
        f"{'model':14s} {'P ours/paper':>14s} {'R ours/paper':>14s} {'F1 ours/paper':>15s}"
        f" {'country strict/match/paper':>27s}"
    )
    for r in rows:
        country = "-"
        if r["country_paper"]:
            match = r["country_match_ours"]
            match = "n/a" if match is None else f"{match:.1f}"
            country = f"{r['country_ours']:.1f}/{match}/{r['country_paper']:.1f}"
        print(
            f"{r['model']:14s} {r['precision_ours']:6.2f}/{r['precision_paper']:<6.2f}"
            f" {r['recall_ours']:6.2f}/{r['recall_paper']:<6.2f}"
            f" {r['f1_ours']:6.2f}/{r['f1_paper']:<7.2f} {country:>27s}"
        )
    noise = load_summaries(scores, noise_tag).get("gemma3_12b")
    if noise:
        base = summaries["gemma3_12b"]["f1"]
        print(
            f"judge noise, gemma3_12b F1: {tag} {base:.2f} vs {noise_tag} {noise['f1']:.2f}"
            f" (diff {noise['f1'] - base:+.2f})"
        )
    f1_checks = check_f1(summaries)
    show(f1_checks)
    verdicts = {}
    for key in ("country_acc", "country_acc_match"):
        if not all(key in summaries[m] for m in OPEN_MODELS):
            continue
        country_checks = check_country(summaries, key)
        show(country_checks)
        verdicts[key] = all(c.passed for c in f1_checks + country_checks)
    for key, passed in verdicts.items():
        print(f"G1 with {key}: {'PASS' if passed else 'FAIL'}")
    print(f"(table written to {out})")
    return any(verdicts.values())


def show(checks) -> None:
    for check in checks:
        print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True, help="scores dir from PACE")
    parser.add_argument("--tag", default="paper_s0")
    parser.add_argument("--noise-tag", default="paper_s1")
    parser.add_argument("--out", type=Path, default=Path("results/validation.csv"))
    args = parser.parse_args()
    report_validation(args.validation, args.tag, args.noise_tag, args.out)


if __name__ == "__main__":
    main()
