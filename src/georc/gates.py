"""Gate G1 checks against the paper's Table 2 (v2) and Table 6; missing values raise KeyError."""

from collections.abc import Mapping
from dataclasses import dataclass

_EPS = 1e-9  # so a value exactly on a tolerance edge is not failed by float rounding


@dataclass(frozen=True)
class PaperRow:
    precision: float
    recall: float
    f1: float
    f1_ci: tuple[float, float] | None  # Table 6, 95% CI
    country: float | None  # country accuracy, %


TABLE2: dict[str, PaperRow] = {
    "qwen": PaperRow(23.95, 48.26, 29.09, (28.03, 30.15), 69.40),
    "gemma3_12b": PaperRow(23.55, 35.55, 25.23, (24.05, 26.42), 64.00),
    "llama3_11b": PaperRow(22.26, 36.86, 24.00, (22.86, 25.14), 58.40),
    "qwen3": PaperRow(26.78, 25.15, 22.41, (21.17, 23.65), 64.40),
    "hallucinated": PaperRow(27.91, 20.12, 18.13, None, None),
}
OPEN_MODELS = ("qwen", "gemma3_12b", "llama3_11b", "qwen3")  # paper's F1 order, best first


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def check_f1(summaries: Mapping[str, Mapping[str, float]]) -> list[Check]:
    checks = []
    for model in OPEN_MODELS:
        low, high = TABLE2[model].f1_ci
        low, high = low - 1.0, high + 1.0
        f1 = summaries[model]["f1"]
        checks.append(
            Check(
                f"{model} F1 within paper CI +/- 1.0",
                low - _EPS <= f1 <= high + _EPS,
                f"ours {f1:.2f}, allowed [{low:.2f}, {high:.2f}]",
            )
        )
    order = tuple(sorted(OPEN_MODELS, key=lambda m: summaries[m]["f1"], reverse=True))
    checks.append(Check("F1 ranking matches the paper", order == OPEN_MODELS, " > ".join(order)))
    f1 = summaries["hallucinated"]["f1"]
    checks.append(
        Check(
            "hallucinated F1 within 2 points of 18.13",
            abs(f1 - TABLE2["hallucinated"].f1) <= 2.0 + _EPS,
            f"ours {f1:.2f}",
        )
    )
    return checks


def check_country(summaries: Mapping[str, Mapping[str, float]], key: str) -> list[Check]:
    """`key`: "country_acc" (strict) or "country_acc_match" (the judge sees the true country)."""
    checks = []
    for model in OPEN_MODELS:
        ours, paper = summaries[model][key], TABLE2[model].country
        checks.append(
            Check(
                f"{model} country accuracy within 3 points",
                abs(ours - paper) <= 3.0 + _EPS,
                f"{key}: ours {ours:.1f}%, paper {paper:.1f}%",
            )
        )
    return checks


def validation_rows(summaries: Mapping[str, Mapping[str, float]]) -> list[dict]:
    """One row per released chain set: ours next to the paper's."""
    rows = []
    for model in (*OPEN_MODELS, "hallucinated"):
        ours, paper = summaries[model], TABLE2[model]
        rows.append(
            {
                "model": model,
                "precision_ours": ours["precision"],
                "precision_paper": paper.precision,
                "recall_ours": ours["recall"],
                "recall_paper": paper.recall,
                "f1_ours": ours["f1"],
                "f1_paper": paper.f1,
                "f1_delta": ours["f1"] - paper.f1,
                "country_ours": ours.get("country_acc"),
                "country_match_ours": ours.get("country_acc_match"),
                "country_paper": paper.country,
            }
        )
    return rows
