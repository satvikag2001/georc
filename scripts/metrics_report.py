"""Alternative metrics from the judge caches pulled from PACE (pace/pull.sh); "-" marks a number
whose cache is missing or incomplete."""

import argparse
import csv
from pathlib import Path

from georc.extras import EXPERTS, PAPER_EXPERTS
from georc.metrics import (
    DISTANCES,
    LOWER_IS_BETTER,
    MODELS,
    collect,
    rescale,
    summary_line,
    value,
)
from georc.stats import paired_bootstrap

PAPER = "paper"  # GeoRC Table 2 "Human Expert Average"
PAIR_ROWS = [(EXPERTS, "shared")] + [(m, s) for m in MODELS for s in ("all", "shared")]
CHAIN_ROWS = [(EXPERTS, "all"), (EXPERTS, "shared")] + PAIR_ROWS[1:]
TABLES = [
    (
        "Soft one-to-all F1 (repo rubric); rescaled: hallucinated 0, experts 100 (shared rounds)",
        [(EXPERTS, "shared"), (EXPERTS, "reverse"), (PAPER, "Table2"), *PAIR_ROWS[1:]],
        [("n", "n_pairs", 0), ("P", "P", 2), ("R", "R", 2), ("F1", "F1", 2)]
        + [("rescaled", "F1_rescaled", 1)],
    ),
    (
        "Matching and gating",
        PAIR_ROWS,
        [("hard50", "hard_F1_t50", 2), ("hard70", "hard_F1_t70", 2)]
        + [("1:1 P", "o2o_P", 2), ("1:1 R", "o2o_R", 2), ("1:1 F1", "o2o_F1", 2)]
        + [("gated", "gated_F1", 2), ("exp win%", "win_rate", 1)]
        + [("lo", "win_rate_lo", 1), ("hi", "win_rate_hi", 1)],
    ),
    (
        "Specificity and grounding",
        PAIR_ROWS,
        [("spec", "spec_mean", 1), ("spec R", "wspec_R", 2), ("spec F1", "wspec_F1", 2)]
        + [("grnd%", "grounded_rate", 1), ("novis%", "no_visual_share", 1)]
        + [("unpars%", "verdict_unparsed", 1), ("grnd P", "grounded_P", 2)]
        + [("grnd F1", "grounded_F1", 2)],
    ),
    (
        "Location, per chain (median km; % within; median km when the country is right/wrong)",
        CHAIN_ROWS,
        [("n", "n_chains", 0), ("noguess%", "no_guess", 1), ("km", "median_km", 0)]
        + [(f"<={d}", f"within_{d}", 1) for d in DISTANCES]
        + [("km|right", "median_km_right", 0), ("km|wrong", "median_km_wrong", 0)],
    ),
]
SUMMARY = [
    ("F1", "soft F1"),
    ("hard_F1_t50", "hard F1 t=50"),
    ("hard_F1_t70", "hard F1 t=70"),
    ("o2o_F1", "one-to-one F1"),
    ("gated_F1", "gated F1"),
    ("win_rate", "expert win %"),
    ("spec_mean", "specificity"),
    ("wspec_R", "spec-weighted R"),
    ("wspec_F1", "spec-weighted F1"),
    ("grounded_rate", "grounded %"),
    ("no_visual_share", "no visual claim %"),
    ("grounded_P", "grounded P"),
    ("grounded_F1", "grounded F1"),
    ("no_guess", "no guess %"),
    ("median_km", "median km"),
    *((f"within_{d}", f"within {d} km %") for d in DISTANCES),
]
SHORT = {"qwen": "qwen", "gemma3_12b": "gemma", "llama3_11b": "llama", "qwen3": "qwen3"}
SHORT |= {"G0": "G0", "A1": "A1", "hallucinated": "halluc"}
WIDTH = 9


def numbers(collected: dict) -> dict[tuple[str, str, str], float]:
    """Every printed number by (set, scope, metric), including counts, intervals and rescaling."""
    found = {}
    for name, scopes in collected.items():
        for scope, metrics in scopes.items():
            for metric in metrics:
                found[name, scope, metric] = value(collected, name, scope, metric)
            for metric, count in (("F1", "n_pairs"), ("median_km", "n_chains")):
                if metric in metrics:
                    found[name, scope, count] = sum(map(len, metrics[metric].values()))
            if "win_rate" in metrics:
                _, low, high = paired_bootstrap(metrics["win_rate"])
                found[name, scope, "win_rate_lo"], found[name, scope, "win_rate_hi"] = low, high
    e, h = found.get((EXPERTS, "shared", "F1")), found.get(("hallucinated", "shared", "F1"))
    for m in MODELS:
        rescaled = rescale(found.get((m, "shared", "F1")), h, e)
        if rescaled is not None:
            found[m, "shared", "F1_rescaled"] = rescaled
    found[PAPER, "Table2", "n_pairs"] = 450
    for key, metric in (("precision", "P"), ("recall", "R"), ("f1", "F1")):
        found[PAPER, "Table2", metric] = PAPER_EXPERTS[key]
    return found


def cell(v: float | str | None, digits: int = 2, width: int = WIDTH) -> str:
    if v is None:
        return f"{'-':>{width}s}"
    if isinstance(v, str):
        return f"{v:>{width}s}"
    return f"{v:>{width}.{digits}f}"


def print_table(title: str, rows: list, columns: list, found: dict) -> None:
    print(f"\n{title}")
    print(f"{'set':13s}{'scope':8s}" + "".join(f"{h:>{WIDTH}s}" for h, _, _ in columns))
    for name, scope in rows:
        cells = [cell(found.get((name, scope, key)), digits) for _, key, digits in columns]
        print(f"{name:13s}{scope:8s}" + "".join(cells))


def print_summary(lines: dict[str, dict]) -> None:
    models = [m for m in MODELS if m != "hallucinated"]
    print(
        "\nSummary, shared rounds (* sign flipped so higher is better; pos = 100 (M - H) / (E - H);"
        " +- = 95% bootstrap half-width over rounds)"
    )
    header = f"{'metric':20s}{'E':>{WIDTH}s}{'H':>{WIDTH}s}{'weakest':>15s}{'strongest':>15s}"
    header += "".join(f"{SHORT[m]:>7s}" for m in models)
    print(header + f"{'+-E-top':>{WIDTH}s}{'+-low-H':>{WIDTH}s}{'order':>6s}")
    for metric, label in SUMMARY:
        line = lines[metric]
        ends = [
            f"{SHORT[line[k]]} {line[k + '_value']:.1f}" if line[k] else None
            for k in ("weakest", "strongest")
        ]
        ordered = None if line["ordered"] is None else ("yes" if line["ordered"] else "no")
        print(
            f"{label + (' *' if metric in LOWER_IS_BETTER else ''):20s}"
            + cell(line["E"], 1)
            + cell(line["H"], 1)
            + "".join(cell(end, width=15) for end in ends)
            + "".join(cell(line[f"pos_{m}"], 0, 7) for m in models)
            + cell(line["hw_E_minus_strongest"], 1)
            + cell(line["hw_weakest_minus_H"], 1)
            + cell(ordered, width=6)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True, help="e.g. runs/georc")
    parser.add_argument("--out", type=Path, default=Path("results/metrics.csv"))
    args = parser.parse_args()

    collected = collect(args.data, args.runs)
    found = numbers(collected)
    for title, rows, columns in TABLES:
        print_table(title, rows, columns, found)
    print(
        "experts shared = the paper's 450 pairs (candidate & reference 1&2, 2&3, 1&3);"
        " reverse = the other direction; shared = the 150 rounds with three experts"
    )
    lines = {metric: summary_line(collected, metric) for metric, _ in SUMMARY}
    print_summary(lines)

    rows = [
        {"section": "metric", "set": name, "scope": scope, "metric": metric, "value": v}
        for (name, scope, metric), v in found.items()
        if v is not None
    ]
    for metric, line in lines.items():
        rows += [
            {"section": "summary", "set": field, "scope": "shared", "metric": metric, "value": v}
            for field, v in line.items()
            if v is not None
        ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "set", "scope", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n(written to {args.out})")


if __name__ == "__main__":
    main()
