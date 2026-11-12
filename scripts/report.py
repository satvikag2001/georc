"""Print gate G1 tables and verdicts from score summaries pulled from PACE (pace/pull.sh)."""

import argparse
from pathlib import Path

from georc import authors
from georc.evaluate import load_summaries, write_rows
from georc.gates import OPEN_MODELS, TABLE2, check_country, check_f1, validation_rows


def report_validation(scores: Path, tag: str, noise_tag: str, out: Path) -> None:
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


def report_authors(run_dir: Path) -> None:
    """The authors' scorer next to Table 2 of arXiv v2 and v1; partial runs are fine.

    Second table: their F1 recomputed from the saved replies under each reading, and against
    Human_Expert_3 only (the other reading of the Human_Chain_3.txt their code expects).
    """
    heads = ("their code P/R/F1", "Table 2 v2", "Table 2 v1")
    print(f"{'model':14s} {'n':>4s} " + " ".join(f"{h:>20s}" for h in heads))
    found = {}
    for tag in authors.TAGS:
        rows = authors.load_pairs(run_dir, authors.completed_units(run_dir, tag))
        if not rows:
            continue
        found[tag] = rows
        ours, paper, board = authors.summarize(rows), TABLE2[tag], authors.TABLE2_V1[tag]
        triple = "/".join(f"{x:.2f}" for x in (ours["precision"], ours["recall"], ours["f1"]))
        print(
            f"{tag:14s} {ours['n']:4d} {triple:>20s}"
            f" {f'{paper.precision:.2f}/{paper.recall:.2f}/{paper.f1:.2f}':>20s}"
            f" {'/'.join(f'{x:.2f}' for x in board):>20s}"
        )
    print(
        f"\n{'model':14s} {'F1 CI95 ours':>13s} {'paper':>13s}  "
        + " ".join(f"{r:>12s}" for r in authors.READINGS)
        + f" {'Expert_3 only':>15s}"
    )
    for tag, rows in found.items():
        low, high = authors.summarize(rows)["f1_ci95"]
        ci = TABLE2[tag].f1_ci
        paper_ci = "-" if ci is None else f"{ci[0]:.2f}-{ci[1]:.2f}"
        by_reading = [
            authors.summarize([authors.pair_metrics(r["calls"], k) for r in rows])["f1"]
            for k in authors.READINGS
        ]
        third = [r for r in rows if r["expert"] == "Human_Expert_3"]
        expert3 = f"{authors.summarize(third)['f1']:.2f} (n={len(third)})" if third else "-"
        print(
            f"{tag:14s} {f'{low:.2f}-{high:.2f}':>13s} {paper_ci:>13s}  "
            + " ".join(f"{f1:12.2f}" for f1 in by_reading)
            + f" {expert3:>15s}"
        )


def show(checks) -> None:
    for check in checks:
        print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, help="scores dir from PACE")
    parser.add_argument(
        "--authors",
        type=Path,
        help="run dir of the authors' scorer, e.g. runs/georc/authors/float32_s0",
    )
    parser.add_argument("--tag", default="paper_s0")
    parser.add_argument("--noise-tag", default="paper_s1")
    parser.add_argument("--out", type=Path, default=Path("results/validation.csv"))
    args = parser.parse_args()
    if not (args.validation or args.authors):
        parser.error("give --validation and/or --authors")
    if args.validation:
        report_validation(args.validation, args.tag, args.noise_tag, args.out)
    if args.authors:
        report_authors(args.authors)


if __name__ == "__main__":
    main()
