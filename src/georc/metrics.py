"""Alternative metrics from the cached judge passes, for the local report after pace/pull.sh.

collect() gathers each metric's per-pair (or per-chain) values by chain set, scope and round;
value() and summary_line() aggregate them. A metric whose cache is missing or incomplete is left
out, so it reports as missing. Not imported by any PACE job (it needs scipy).
"""

import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import fmean, median
from typing import Any

from scipy.optimize import linear_sum_assignment

from georc.data import Pair, load_coordinates, load_pairs, load_rounds
from georc.extras import (
    EXPERTS,
    GENERATED,
    RELEASED,
    VARIANT,
    CacheMiss,
    CacheOnly,
    Chain,
    cache_path,
    load_set,
    paper_expert_pair,
    run_pass,
)
from georc.gates import OPEN_MODELS
from georc.judge import CachedJudge
from georc.points import extract_points
from georc.score import pair_score, point_scores
from georc.stats import bootstrap_difference

MODELS = (*RELEASED, *GENERATED)
THRESHOLDS = tuple(range(30, 100, 10))
DISTANCES = (25, 200, 750, 2500)
EARTH_RADIUS_KM = 6371.0
LOWER_IS_BETTER = frozenset({"win_rate", "no_visual_share", "no_guess", "median_km"})
_IDS = ("round", "author", "expert")
_MISSING = (CacheMiss, FileNotFoundError, json.JSONDecodeError)


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def _zeros(scores: Sequence[float | None]) -> list[float]:
    return [0.0 if s is None else s for s in scores]


def hard_f1(candidate: Sequence[float | None], expert: Sequence[float | None], t: float) -> float:
    """F1 of the shares of candidate and expert points whose one-to-all score is at least t;
    unparsed scores never match."""
    if not candidate or not expert:
        return 0.0
    precision = 100 * sum(s is not None and s >= t for s in candidate) / len(candidate)
    recall = 100 * sum(s is not None and s >= t for s in expert) / len(expert)
    return f1(precision, recall)


def one_to_one(matrix: Sequence[Sequence[float | None]]) -> tuple[float, float, float]:
    """P, R, F1 of the maximum-weight one-to-one matching of expert points (rows) to candidate
    points (columns): the matched total over the candidate and over the expert point counts.
    Unparsed cells count 0."""
    if not matrix or not matrix[0]:
        return 0.0, 0.0, 0.0
    weights = [_zeros(row) for row in matrix]
    rows, cols = linear_sum_assignment(weights, maximize=True)
    total = float(sum(weights[i][j] for i, j in zip(rows, cols, strict=True)))
    precision, recall = total / len(matrix[0]), total / len(matrix)
    return precision, recall, f1(precision, recall)


def weighted_recall(
    expert_scores: Sequence[float | None], weights: Sequence[float | None]
) -> float:
    """Recall with each expert point weighted by its specificity; an unparsed score or weight
    counts 0, and if every weight is 0 the points weigh equally (plain recall)."""
    scores, w = _zeros(expert_scores), _zeros(weights)
    if not any(w):
        return fmean(scores)
    return sum(s * x for s, x in zip(scores, w, strict=True)) / sum(w)


def grounded_precision(scores: Sequence[float | None], verdicts: Sequence[str | None]) -> float:
    """Mean candidate-point score, counting 0 where the vision judge said "no"; an unparsed
    verdict counts as not "no"."""
    return fmean(0.0 if v == "no" else s for s, v in zip(_zeros(scores), verdicts, strict=True))


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance between two (lat, lon) points in degrees."""
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2
    h += math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, h)))


def beat_rate(rivals: Sequence[float], own: float) -> float:
    """Share of rival scores above `own`, ties counting half."""
    return fmean(1.0 if r > own else 0.5 if r == own else 0.0 for r in rivals)


def rescale(m: float | None, h: float | None, e: float | None) -> float | None:
    """100 * (m - h) / (e - h): hallucinated at 0, experts at 100."""
    if m is None or h is None or e is None or e == h:
        return None
    return 100 * (m - h) / (e - h)


def win_units(
    set_f1: dict[tuple[str, str, str], float], expert_f1: dict[tuple[str, str, str], float]
) -> dict[str, list[float]]:
    """Per round, one value per reference expert: the chance (%) that another expert's F1 against
    that reference beats the set's, ties counting half, averaged over the other experts (and over
    the set's candidates). Keys are (round, candidate author, reference expert)."""
    experts: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for (rnd, author, ref), score in expert_f1.items():
        experts[rnd, ref][author] = score
    units: dict[str, list[float]] = defaultdict(list)
    candidates: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for (rnd, author, ref), score in set_f1.items():
        if (rnd, ref) in experts:
            candidates[rnd, ref][author] = score
    for (rnd, ref), own in candidates.items():
        rivals = experts[rnd, ref]
        rates = [
            beat_rate([s for a, s in rivals.items() if a != author], score)
            for author, score in own.items()
            if any(a != author for a in rivals)
        ]
        if rates:
            units[rnd].append(100 * fmean(rates))
    return dict(units)


def pair_records(
    pairs: Sequence[tuple[Chain, Pair]],
    points: Sequence[tuple[list[float | None], list[float | None]]] | None,
    pairwise: dict[tuple[str, str, str], list[list[float | None]]] | None = None,
    candidate_spec: dict[str, float | None] | None = None,
    expert_spec: dict[str, float | None] | None = None,
    verdicts: dict[tuple[str, str], list[str | None]] | None = None,
    correct: dict[tuple[str, str], bool] | None = None,
) -> list[dict]:
    """Per-pair metric values; a metric whose source is None is left out of every record.

    `points` are the one-to-all point scores; the other sources are keyed by (round, author[,
    expert]); point lists line up because every pass splits chains with extract_points.
    """
    records = []
    for i, (chain, ref) in enumerate(pairs):
        rnd, author = chain.round.key, chain.author
        record: dict[str, Any] = {"round": rnd, "author": author, "expert": ref.expert}
        if points is not None:
            cand, gt = points[i]
            score = pair_score(cand, gt)
            record.update(P=score.precision, R=score.recall, F1=score.f1)
            record.update({f"hard_F1_t{t}": hard_f1(cand, gt, t) for t in THRESHOLDS})
            if correct is not None:
                record["gated_F1"] = score.f1 * correct[rnd, author]
            if expert_spec is not None:
                weights = [expert_spec[g] for g in extract_points(ref.chain)]
                recall = weighted_recall(gt, weights) if gt else 0.0
                record.update(wspec_R=recall, wspec_F1=f1(score.precision, recall))
            if verdicts is not None:
                precision = grounded_precision(cand, verdicts[rnd, author]) if cand else 0.0
                record.update(grounded_P=precision, grounded_F1=f1(precision, score.recall))
        if pairwise is not None:
            o2o = one_to_one(pairwise[rnd, author, ref.expert])
            record.update(zip(("o2o_P", "o2o_R", "o2o_F1"), o2o, strict=True))
        if candidate_spec is not None:
            rated = [candidate_spec[p] for p in extract_points(chain.text)]
            rated = [s for s in rated if s is not None]
            record["spec_mean"] = fmean(rated) if rated else None
        if verdicts is not None:
            found = verdicts[rnd, author]
            yes, no = found.count("yes"), found.count("no")
            record["grounded_rate"] = 100 * yes / (yes + no) if yes + no else None
            record["no_visual_share"] = 100 * found.count("na") / len(found) if found else None
            record["verdict_unparsed"] = 100 * found.count(None) / len(found) if found else None
        records.append(record)
    return records


def chain_records(
    locations: list[dict],
    truth: dict[str, tuple[float, float]],
    correct: dict[tuple[str, str], bool] | None = None,
) -> list[dict]:
    """Per chain whose location reply parsed: the great-circle error (inf without coordinates),
    the no-guess and within-distance flags (0 or 100), and the error again under
    median_km_right or median_km_wrong by the strict country check."""
    records = []
    for row in locations:
        if not row["parsed"]:
            continue
        located = row["lat"] is not None
        km = haversine_km((row["lat"], row["lon"]), truth[row["round"]]) if located else math.inf
        record = {"round": row["round"], "author": row["author"], "median_km": km}
        record["no_guess"] = 0.0 if located else 100.0
        record.update({f"within_{d}": 100.0 * (km <= d) for d in DISTANCES})
        if correct is not None:
            right = correct[row["round"], row["author"]]
            record["median_km_right" if right else "median_km_wrong"] = km
        records.append(record)
    return records


def items(records: list[dict], rounds: set[str] | None = None) -> dict[str, dict[str, list]]:
    """Metric -> round -> the records' values (None skipped), optionally only for `rounds`."""
    found: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if rounds is None or record["round"] in rounds:
            for key, v in record.items():
                if key not in _IDS and v is not None:
                    found[key][record["round"]].append(v)
    return {key: dict(by_round) for key, by_round in found.items()}


def statistic(metric: str) -> Callable[[list[float]], float]:
    return median if metric.startswith("median_km") else fmean


def value(collected: dict, name: str, scope: str, metric: str) -> float | None:
    by_round = collected.get(name, {}).get(scope, {}).get(metric)
    if not by_round:
        return None
    return statistic(metric)([v for values in by_round.values() for v in values])


def _try(load: Callable[..., Any], *args: Any) -> Any:
    """`load(*args)`, or None when a cache it reads is missing or incomplete."""
    try:
        return load(*args)
    except _MISSING:
        return None


def _rounds_correct(runs: Path, name: str) -> dict[tuple[str, str], bool] | None:
    """Strict country correctness from score_chains.py's rounds.csv (repo rubric, seed 0)."""
    path = runs / "scores" / f"{name}__repo_s0" / "rounds.csv"
    if not path.exists():
        return None
    with path.open() as f:
        return {(row["round"], name): row["correct"] == "1" for row in csv.DictReader(f)}


def collect(data: Path, runs: Path) -> dict[str, dict[str, dict[str, dict[str, list]]]]:
    """set -> scope -> metric -> round -> values, from the caches under `runs`.

    Model sets: scope "all" (800 pairs, 500 chains) and "shared" (the 150 rounds with three
    experts: 450 pairs, 150 chains). Experts: "shared" is the paper's 450 pairs and the 450 expert
    chains of those rounds, "reverse" the other 450 pairs, "all" every expert chain (location).
    """
    rounds = load_rounds(data)
    counts = Counter(p.round.key for p in load_pairs(data, rounds))
    shared = {key for key, n in counts.items() if n > 1}
    truth = load_coordinates(data)

    def reader(path: Path) -> CachedJudge | None:
        # A missing file is checked first: CachedJudge would create its folder under runs/.
        return _try(CachedJudge, CacheOnly(), path) if path.exists() else None

    def cached(pass_name: str, name: str) -> Any:
        """A pass's results re-read from its cache, or None if that is missing or incomplete."""
        judge = reader(cache_path(runs, pass_name, name))
        return None if judge is None else _try(run_pass, pass_name, name, data, runs, judge)

    repo = reader(runs / "judge_cache" / "repo_s0.jsonl")
    experts = cached("experts", EXPERTS)
    expert_spec = cached("specificity", EXPERTS)
    expert_f1 = None
    if experts is not None:
        expert_f1 = {(r["round"], r["author"], r["expert"]): r["f1"] for r in experts[0]}
    collected = {}
    for name in (EXPERTS, *MODELS):
        try:
            _, pairs = load_set(name, data, runs)
        except FileNotFoundError:
            continue
        if name == EXPERTS:
            judge = reader(cache_path(runs, "experts", EXPERTS))
            correct = None
            if experts is not None:
                correct = {(r["round"], r["author"]): r["correct"] == 1 for r in experts[1]}
        else:
            judge, correct = repo, _rounds_correct(runs, name)
        texts = [(c.text, p.chain) for c, p in pairs]
        points = None if judge is None else _try(point_scores, texts, judge, VARIANT)
        pairwise = cached("pairwise", name)
        if pairwise is not None:
            pairwise = {(r["round"], r["author"], r["expert"]): r["scores"] for r in pairwise}
        grounded = cached("grounded", name)
        if grounded is not None:
            grounded = {(r["round"], r["author"]): r["verdicts"] for r in grounded}
        location = cached("location", name)
        records = pair_records(
            pairs,
            points,
            pairwise=pairwise,
            candidate_spec=cached("specificity", name),
            expert_spec=expert_spec,
            verdicts=grounded,
            correct=correct,
        )
        chains = [] if location is None else chain_records(location, truth, correct)
        wins = {}
        if points is not None and expert_f1 is not None:
            set_f1 = {(r["round"], r["author"], r["expert"]): r["F1"] for r in records}
            wins = {"win_rate": win_units(set_f1, expert_f1)}
        if name == EXPERTS:
            paper = [r for r in records if paper_expert_pair(r["author"], r["expert"])]
            reverse = [r for r in records if not paper_expert_pair(r["author"], r["expert"])]
            collected[name] = {
                "shared": {**items(paper), **items(chains, shared), **wins},
                "reverse": items(reverse),
                "all": items(chains),
            }
        else:
            collected[name] = {
                "all": {**items(records), **items(chains)},
                "shared": {**items(records, shared), **items(chains, shared), **wins},
            }
    return collected


def _half_width(groups: dict, a: str | None, b: str | None, stat: Callable) -> float | None:
    if a not in groups or b not in groups:
        return None
    _, low, high = bootstrap_difference(groups[a], groups[b], stat)
    return (high - low) / 2


def summary_line(collected: dict, metric: str) -> dict[str, Any]:
    """Experts (E), hallucinated (H), the weakest and strongest released model, every model's
    position 100 (M - H) / (E - H), 95% bootstrap half-widths of E - strongest and weakest - H
    (rounds resampled), and whether experts > every model > hallucinated; shared rounds only,
    with the sign flipped for lower-is-better metrics so that higher is always better."""
    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
    stat = statistic(metric)
    groups = {}
    for name in (EXPERTS, *MODELS):
        by_round = collected.get(name, {}).get("shared", {}).get(metric)
        if by_round:
            groups[name] = {r: [sign * v for v in values] for r, values in by_round.items()}
    values = {name: stat([v for vs in g.values() for v in vs]) for name, g in groups.items()}
    e, h = values.get(EXPERTS), values.get("hallucinated")
    released = [m for m in OPEN_MODELS if m in values]
    weakest = min(released, key=values.__getitem__) if released else None
    strongest = max(released, key=values.__getitem__) if released else None
    others = [values.get(m) for m in MODELS if m != "hallucinated"]
    line = {
        "E": e,
        "H": h,
        "weakest": weakest,
        "weakest_value": values.get(weakest),
        "strongest": strongest,
        "strongest_value": values.get(strongest),
        **{f"pos_{m}": rescale(values.get(m), h, e) for m in MODELS if m != "hallucinated"},
        "hw_E_minus_strongest": _half_width(groups, EXPERTS, strongest, stat),
        "hw_weakest_minus_H": _half_width(groups, weakest, "hallucinated", stat),
        "ordered": None,
    }
    if e is not None and h is not None and None not in others:
        line["ordered"] = all(e > v > h for v in others)
    return line
