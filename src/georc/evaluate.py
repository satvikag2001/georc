"""Score a set of chains (released or generated) against the expert chains with a judge.

A "chain set" maps each round key ("<challenge>_<index>") to the chain text for that round.
Pair rows feed F1; round rows feed country accuracy.
"""

import csv
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

from georc.country import build_country_prompt, country_accuracy, parse_country
from georc.data import Pair, Round, released_chain_path
from georc.judge import Judge
from georc.score import score_pairs, summarize


def released_chain_set(root: Path, rounds: list[Round], tag: str) -> dict[str, str]:
    return {r.key: released_chain_path(root, r, tag).read_text() for r in rounds}


def directory_chain_set(folder: Path, rounds: list[Round]) -> dict[str, str]:
    """Generated chains stored as <folder>/<round key>.txt; missing rounds are an error."""
    missing = [r.key for r in rounds if not (folder / f"{r.key}.txt").exists()]
    if missing:
        raise FileNotFoundError(f"{folder}: {len(missing)} chains missing, e.g. {missing[:3]}")
    return {r.key: (folder / f"{r.key}.txt").read_text() for r in rounds}


def evaluate(
    chains: Mapping[str, str],
    pairs: list[Pair],
    rounds: list[Round],
    judge: Judge,
    variant: str = "paper",
) -> tuple[list[dict], list[dict], dict]:
    """Return (pair rows, round rows, summary) for one chain set."""
    scores = score_pairs([(chains[p.round.key], p.chain) for p in pairs], judge, variant)
    pair_rows = [
        {"round": p.round.key, "expert": p.expert, **asdict(s)}
        for p, s in zip(pairs, scores, strict=True)
    ]
    replies = judge.complete([build_country_prompt(chains[r.key]) for r in rounds])
    guesses = [parse_country(reply) for reply in replies]
    round_rows = [
        {"round": r.key, "truth": r.country, "guess": g or "", "correct": int(g == r.country)}
        for r, g in zip(rounds, guesses, strict=True)
    ]
    summary = {
        **summarize(scores),
        "country_acc": country_accuracy(guesses, [r.country for r in rounds]),
        "country_unparsed": sum(g is None for g in guesses),
        "mean_points": fmean(s.n_candidate_points for s in scores),
    }
    return pair_rows, round_rows, summary


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
