"""Turn score summaries pulled from PACE into result tables and gate verdicts.

bash pace/pull.sh && report.py --validation runs/georc/scores   # gate G1 on released chains
"""

import argparse
from pathlib import Path

from georc.evaluate import load_summaries, write_rows
from georc.gates import check_g1, validation_rows


def report_validation(scores: Path, tag: str, noise_tag: str, out: Path) -> bool:
    summaries = load_summaries(scores, tag)
    rows = validation_rows(summaries)
    write_rows(out, rows)
    print(
        f"{'model':14s} {'P ours/paper':>14s} {'R ours/paper':>14s} {'F1 ours/paper':>15s}"
        f" {'country ours/paper':>19s}"
    )
    for r in rows:
        country = f"{r['country_ours']:.1f}/{r['country_paper']:.1f}" if r["country_paper"] else "-"
        print(
            f"{r['model']:14s} {r['precision_ours']:6.2f}/{r['precision_paper']:<6.2f}"
            f" {r['recall_ours']:6.2f}/{r['recall_paper']:<6.2f}"
            f" {r['f1_ours']:6.2f}/{r['f1_paper']:<7.2f} {country:>19s}"
        )
    noise = load_summaries(scores, noise_tag).get("gemma3_12b")
    if noise:
        base = summaries["gemma3_12b"]["f1"]
        print(
            f"judge noise, gemma3_12b F1: {tag} {base:.2f} vs {noise_tag} {noise['f1']:.2f}"
            f" (diff {noise['f1'] - base:+.2f})"
        )
    checks = check_g1(summaries)
    for check in checks:
        print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    passed = all(c.passed for c in checks)
    print(f"G1: {'PASS' if passed else 'FAIL'}  (table written to {out})")
    return passed


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
