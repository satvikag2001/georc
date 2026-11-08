"""One-to-all precision, recall and F1 between candidate and expert chains (GeoRC Algorithm 1)."""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean

from georc.judge import Judge, build_prompt, parse_score
from georc.points import extract_points


@dataclass(frozen=True)
class PairScore:
    precision: float  # 0-100
    recall: float
    f1: float
    n_candidate_points: int
    n_expert_points: int
    n_unparsed: int


def score_pairs(
    pairs: Sequence[tuple[str, str]], judge: Judge, variant: str = "paper"
) -> list[PairScore]:
    """Score (candidate chain, expert chain) pairs, sending every judge call in one batch.

    Precision: each candidate point against the whole expert chain (points joined by " || ").
    Recall: each expert point against the whole candidate chain. A pair with no points on either
    side scores 0/0/0, as in the released code.
    """
    prompts: list[str] = []
    plans: list[tuple[int, int, int] | None] = []  # (offset, n_candidate, n_expert)
    for candidate, expert in pairs:
        cand_points, gt_points = extract_points(candidate), extract_points(expert)
        if not cand_points or not gt_points:
            plans.append(None)
            continue
        plans.append((len(prompts), len(cand_points), len(gt_points)))
        gt_chain, cand_chain = " || ".join(gt_points), " || ".join(cand_points)
        prompts += [build_prompt(gt_chain, p, variant) for p in cand_points]
        prompts += [build_prompt(cand_chain, g, variant) for g in gt_points]

    parsed = [parse_score(r) for r in judge.complete(prompts)] if prompts else []
    results = []
    for plan in plans:
        if plan is None:
            results.append(PairScore(0.0, 0.0, 0.0, 0, 0, 0))
            continue
        offset, n_cand, n_gt = plan
        window = parsed[offset : offset + n_cand + n_gt]
        scores = [0.0 if s is None else s for s in window]
        precision, recall = fmean(scores[:n_cand]), fmean(scores[n_cand:])
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        unparsed = sum(s is None for s in window)
        results.append(PairScore(precision, recall, f1, n_cand, n_gt, unparsed))
    return results


def summarize(scores: Sequence[PairScore]) -> dict[str, float]:
    """Benchmark numbers: means of per-pair values (the paper's F1 is not F1 of the means)."""
    return {
        "precision": fmean(s.precision for s in scores),
        "recall": fmean(s.recall for s in scores),
        "f1": fmean(s.f1 for s in scores),
        "n": len(scores),
        "unparsed": sum(s.n_unparsed for s in scores),
    }
