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


def point_scores(
    pairs: Sequence[tuple[str, str]], judge: Judge, variant: str = "paper"
) -> list[tuple[list[float | None], list[float | None]]]:
    """Per (candidate chain, expert chain) pair, the judge's score of each candidate point and of
    each expert point, sending every judge call in one batch; ([], []) if a side has no points.

    Precision: each candidate point against the whole expert chain (points joined by " || ").
    Recall: each expert point against the whole candidate chain.
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
        prompts += [build_prompt(cand_chain, g, variant, "recall") for g in gt_points]

    parsed = [parse_score(r) for r in judge.complete(prompts)] if prompts else []
    results = []
    for plan in plans:
        if plan is None:
            results.append(([], []))
            continue
        offset, n_cand, n_gt = plan
        results.append(
            (parsed[offset : offset + n_cand], parsed[offset + n_cand : offset + n_cand + n_gt])
        )
    return results


def pair_score(candidate: list[float | None], expert: list[float | None]) -> PairScore:
    """P, R and F1 from point scores; unparsed scores count 0 and a pair with no points on either
    side scores 0/0/0, as in the released code."""
    if not candidate or not expert:
        return PairScore(0.0, 0.0, 0.0, 0, 0, 0)
    precision = fmean(0.0 if s is None else s for s in candidate)
    recall = fmean(0.0 if s is None else s for s in expert)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    unparsed = sum(s is None for s in candidate + expert)
    return PairScore(precision, recall, f1, len(candidate), len(expert), unparsed)


def score_pairs(
    pairs: Sequence[tuple[str, str]], judge: Judge, variant: str = "paper"
) -> list[PairScore]:
    """Score (candidate chain, expert chain) pairs, sending every judge call in one batch."""
    return [pair_score(c, e) for c, e in point_scores(pairs, judge, variant)]


def summarize(scores: Sequence[PairScore]) -> dict[str, float]:
    """Benchmark numbers: means of per-pair values (the paper's F1 is not F1 of the means)."""
    return {
        "precision": fmean(s.precision for s in scores),
        "recall": fmean(s.recall for s in scores),
        "f1": fmean(s.f1 for s in scores),
        "n": len(scores),
        "unparsed": sum(s.n_unparsed for s in scores),
    }
