"""Pool test: split chains into (cue, location, inference, weight) sets, pool the experts' cues per
image, and score every chain against the pool with closed yes/no judge questions.

Only the rounds written by all three experts are used. Experts are scored leave-one-out (against the
other two); "random" is Human_Expert_1's chain from the next round, which should score near zero.
"""

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from georc.extras import EXPERTS, Chain, build_evidence_prompt, load_set, parse_evidence
from georc.judge import Judge, load_template
from georc.points import extract_points

RANDOM = "random"
MARKS = ("cue", "coverage", "f1", "place", "reason")
_YES_NO = re.compile(r"\W*(yes|no)\b", re.IGNORECASE)
Verdicts = dict[tuple[str, str], bool | None]


@dataclass(frozen=True)
class EvidenceSet:
    round: str
    author: str  # the chain set's name; for experts, the expert's
    cue: str
    location: str | None
    inference: str | None
    weight: float  # share of its chain; a chain's sets sum to 1


def extract_sets(chains: Sequence[Chain], judge: Judge) -> list[EvidenceSet]:
    """Every chain's evidence sets; statements with evidence share the chain's weight equally."""
    plans, prompts = [], []
    for chain in chains:
        points = extract_points(chain.text)
        plans.append((chain, len(prompts), len(points)))
        prompts += [build_evidence_prompt(points, p) for p in points]
    parsed = [parse_evidence(r) for r in judge.complete(prompts)] if prompts else []
    sets = []
    for chain, offset, n in plans:
        found = [s for s in parsed[offset : offset + n] if s]
        for statement in found:
            sets += [
                EvidenceSet(
                    chain.round.key,
                    chain.author,
                    s["cue"],
                    s["location"],
                    s["inference"],
                    s["weight"] / 100 / len(found),
                )
                for s in statement
            ]
    return sets


def ask(
    template: str, pairs: Sequence[tuple[str, str]], judge: Judge, both_orders: bool = False
) -> Verdicts:
    """Yes/no verdict per (first, second) pair, stored under both orders; None if unreadable.
    With `both_orders` the question is asked each way round and a yes needs both."""
    head, rest = load_template(template).split("{a}")
    middle, tail = rest.split("{b}")
    unique = list(dict.fromkeys(pairs))
    asked = unique + [(b, a) for a, b in unique] if both_orders else unique
    replies = judge.complete([head + a + middle + b + tail for a, b in asked]) if asked else []
    found = [_YES_NO.match(reply) for reply in replies]
    answers = [m.group(1).lower() == "yes" if m else None for m in found]
    verdicts: Verdicts = {}
    for i, (a, b) in enumerate(unique):
        verdict = answers[i]
        if both_orders and verdict is not None:
            other = answers[len(unique) + i]
            verdict = None if other is None else verdict and other
        verdicts[(a, b)] = verdicts[(b, a)] = verdict
    return verdicts


def _same(verdicts: Verdicts, a: str, b: str) -> bool:
    return a == b or bool(verdicts.get((a, b)))


def clusters(sets: Sequence[EvidenceSet], same_cue: Verdicts) -> list[list[int]]:
    """Indices of `sets` grouped into pool cues: cues judged the same, transitively."""
    parent = list(range(len(sets)))

    def root(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if _same(same_cue, sets[i].cue, sets[j].cue):
                parent[root(j)] = root(i)
    groups = defaultdict(list)
    for i in range(len(sets)):
        groups[root(i)].append(i)
    return sorted(groups.values())


def _agreement(
    candidate: Sequence[EvidenceSet],
    matches: Sequence[Sequence[EvidenceSet]],
    verdicts: Verdicts,
    text: Any,
) -> float | None:
    """Weighted share of the matched expert sets that agree with the candidate's, 0-100."""
    total = agreed = 0.0
    for s, matched in zip(candidate, matches, strict=True):
        pool = [e for e in matched if text(e) is not None]
        if text(s) is None or not pool:
            continue
        share = sum(e.weight for e in pool if _same(verdicts, text(e), text(s)))
        agreed += s.weight * share / sum(e.weight for e in pool)
        total += s.weight
    return 100 * agreed / total if total else None


def reason_text(s: EvidenceSet) -> str | None:
    return f"{s.cue}: {s.inference}" if s.inference is not None else None


def score_chain(
    candidate: Sequence[EvidenceSet],
    experts: Sequence[EvidenceSet],
    same_cue: Verdicts,
    same_place: Verdicts,
    same_reason: Verdicts,
) -> dict[str, float | None]:
    """Marks (0-100) for one chain against one round's expert sets.

    cue: share of the chain's weight on cues the pool has. coverage: share of the pool's cues the
    chain has. f1: of those two. place / reason: for matched cues, how much of the pool's weight on
    that cue agrees with the chain's place / reason (None when nothing can be compared).
    """
    matches = [[e for e in experts if _same(same_cue, e.cue, s.cue)] for s in candidate]
    cue = 100 * sum(s.weight for s, m in zip(candidate, matches, strict=True) if m)
    groups = clusters(experts, same_cue)
    hit = sum(any(experts[i] in m for m in matches for i in group) for group in groups)
    coverage = 100 * hit / len(groups) if groups else 0.0
    return {
        "cue": cue,
        "coverage": coverage,
        "f1": 2 * cue * coverage / (cue + coverage) if cue + coverage else 0.0,
        "place": _agreement(candidate, matches, same_place, lambda s: s.location),
        "reason": _agreement(candidate, matches, same_reason, reason_text),
    }


def run(
    data: Path,
    out: Path,
    judge: Judge,
    names: Sequence[str],
    cue_template: str = "same_cue.txt",
    both_orders: bool = False,
    match_judge: Judge | None = None,
) -> dict[str, Any]:
    """Extract, match and score the experts, the named chain sets and the random baseline.
    `judge` splits chains into sets; `match_judge` (default: the same) answers the yes/no
    questions."""
    match_judge = match_judge or judge
    expert_chains, _ = load_set(EXPERTS, data, out)
    by_round = defaultdict(list)
    for chain in expert_chains:
        by_round[chain.round.key].append(chain)
    shared = [key for key, chains in by_round.items() if len(chains) == 3]
    chains = {EXPERTS: [c for key in shared for c in by_round[key]]}
    for name in names:
        chains[name] = [c for c in load_set(name, data, out)[0] if c.round.key in shared]

    sets = {name: extract_sets(found, judge) for name, found in chains.items()}
    experts = defaultdict(list)
    for s in sets[EXPERTS]:
        experts[s.round].append(s)
    firsts = {
        key: [s for s in experts[key] if s.author == by_round[key][0].author] for key in shared
    }
    sets[RANDOM] = [
        EvidenceSet(key, RANDOM, s.cue, s.location, s.inference, s.weight)
        for key, other in zip(shared, shared[1:] + shared[:1], strict=True)
        for s in firsts[other]
    ]

    def candidates(name: str) -> dict[tuple[str, str], list[EvidenceSet]]:
        grouped = defaultdict(list)
        for s in sets[name]:
            grouped[(s.round, s.author)].append(s)
        return grouped

    def pairs(text: Any, matched: Verdicts | None) -> list[tuple[str, str]]:
        found = []
        for name in sets:
            for s in sets[name]:
                for e in experts[s.round]:
                    if e is s or (matched is not None and not _same(matched, e.cue, s.cue)):
                        continue
                    a, b = text(e), text(s)
                    if a is not None and b is not None and a != b:
                        found.append((a, b))
        return found

    same_cue = ask(cue_template, pairs(lambda s: s.cue, None), match_judge, both_orders)
    same_place = ask("same_place.txt", pairs(lambda s: s.location, same_cue), match_judge)
    same_reason = ask("same_reason.txt", pairs(reason_text, same_cue), match_judge)

    rows = []
    for name in sets:
        grouped = candidates(name)
        authors = {c.author for c in chains[name]} if name != RANDOM else {RANDOM}
        for key in shared:
            for author in sorted(authors):
                mine = grouped.get((key, author), [])
                pool = [e for e in experts[key] if e.author != author]
                marks = score_chain(mine, pool, same_cue, same_place, same_reason)
                rows.append(
                    {"set": name, "round": key, "author": author, "n_sets": len(mine), **marks}
                )
    summary = {}
    for name in sets:
        mine = [r for r in rows if r["set"] == name]
        summary[name] = {"n": len(mine), "sets_per_chain": fmean(r["n_sets"] for r in mine)} | {
            mark: fmean(v) if (v := [r[mark] for r in mine if r[mark] is not None]) else None
            for mark in MARKS
        }
    return {
        "chains": chains,
        "sets": sets,
        "verdicts": {"cue": same_cue, "place": same_place, "reason": same_reason},
        "rows": rows,
        "summary": summary,
    }


def table(summary: dict[str, dict]) -> str:
    """One line per chain set: marks on a 0-100 scale."""
    lines = [f"{'set':13s} {'chains':>6s} {'sets':>5s} " + " ".join(f"{m:>8s}" for m in MARKS)]
    for name, row in summary.items():
        marks = " ".join(f"{row[m]:8.1f}" if row[m] is not None else f"{'-':>8s}" for m in MARKS)
        lines.append(f"{name:13s} {row['n']:6d} {row['sets_per_chain']:5.1f} {marks}")
    return "\n".join(lines)
