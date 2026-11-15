"""Extra judge passes for alternative metrics, with one reply cache per (pass, chain set).

Each pass builds prompts, sends them through a Judge and parses the replies. On PACE the judge is
the real model behind a CachedJudge (scripts/judge_extras.py), which fills the cache; locally the
same pass re-reads the pulled cache with `CachedJudge(CacheOnly(), cache_path(...))`.
Text passes use the Qwen3-4B judge with seed 0 and the released-code rubric where a rubric applies;
the grounded pass uses Qwen2.5-VL-7B, greedy.
"""

import json
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from georc.country import build_country_prompt, parse_country
from georc.data import Pair, Round, expert_round_texts, load_pairs, load_rounds
from georc.evaluate import directory_chain_set, released_chain_set
from georc.judge import Judge, build_prompt, load_template, parse_score, prompt_key
from georc.points import extract_points
from georc.score import score_pairs

RELEASED = ("qwen", "gemma3_12b", "llama3_11b", "qwen3", "hallucinated")
GENERATED = ("G0", "A1")  # seed-0 chains under <out>/chains/<ID>/seed0
EXPERTS = "experts"
SETS = (*RELEASED, *GENERATED, EXPERTS)
PASSES = ("experts", "pairwise", "specificity", "location", "grounded")
VARIANT = "repo"
GROUNDED_MAX_TOKENS = 512  # about 4 tokens per "N: yes" line; the longest chain has 51 points

# GeoRC v2 Table 2 "Human Expert Average" (N=450) with its Table 6 CI, and the per-expert country
# accuracy rows #1 JD, #2 TS, #3 RC, taken to be Human_Expert_1-3.
PAPER_EXPERTS = {"precision": 55.06, "recall": 65.41, "f1": 56.69, "f1_ci": (55.12, 58.26)}
PAPER_EXPERT_COUNTRY = {"Human_Expert_1": 96.67, "Human_Expert_2": 90.00, "Human_Expert_3": 97.33}

_SPECIFICITY = re.compile(r'"specificity"\s*:\s*(-?\d+(?:\.\d+)?)')
_JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)
_VERDICT = re.compile(
    r"^[\s*-]*(\d+)\s*[:.)]\s*\**\s*(yes|no|n/?a)\b", re.IGNORECASE | re.MULTILINE
)


@dataclass(frozen=True)
class Chain:
    round: Round
    author: str  # the chain set's name; for the experts set, the expert's
    text: str  # the scored chain; for experts the cleaned chain, which stops before the conclusion
    full: str  # the chain with its final guess; for experts the round text with the conclusion


@dataclass(frozen=True)
class Place:
    name: str | None
    lat: float | None
    lon: float | None


def load_set(name: str, data: Path, out: Path) -> tuple[list[Chain], list[tuple[Chain, Pair]]]:
    """A chain set's chains, and each chain paired with every other expert of its round.

    Model sets: 500 chains and the 800 (round, expert) pairs of evaluate.py, in its order.
    Experts: 800 chains and 900 ordered (candidate, reference) pairs on the 150 three-expert rounds.
    """
    rounds = load_rounds(data)
    references = load_pairs(data, rounds)
    if name == EXPERTS:
        texts: dict[Path, dict[int, str]] = {}
        chains = []
        for p in references:
            file = data / p.round.challenge / f"{p.expert}.txt"
            if file not in texts:
                texts[file] = expert_round_texts(file.read_text())
            chains.append(Chain(p.round, p.expert, p.chain, texts[file][p.round.index]))
    else:
        if name in GENERATED:
            found = directory_chain_set(Path(out) / "chains" / name / "seed0", rounds)
        else:
            found = released_chain_set(data, rounds, name)
        chains = [Chain(r, name, found[r.key], found[r.key]) for r in rounds]
    by_round = defaultdict(list)
    for chain in chains:
        by_round[chain.round.key].append(chain)
    pairs = [(c, p) for p in references for c in by_round[p.round.key] if c.author != p.expert]
    return chains, pairs


def paper_expert_pair(author: str, expert: str) -> bool:
    """Whether (candidate, reference) is one of the paper's 450 expert pairs.

    GeoRC Sec. 4.4 compares "expert candidate & reference chains paired as 1 & 2, 2 & 3, and 1 & 3
    over 150 locations each" without saying which is which; read literally, the first named (the
    lower-numbered expert) is the candidate.
    """
    return int(author.rsplit("_", 1)[1]) < int(expert.rsplit("_", 1)[1])


def cache_path(out: Path, pass_name: str, set_name: str) -> Path:
    return Path(out) / "judge_cache" / "extras" / f"{pass_name}__{set_name}.jsonl"


class CacheMiss(LookupError):
    pass


class CacheOnly:
    """Inner judge for re-reading a pulled cache: any prompt that reaches it was never judged."""

    def complete(self, prompts: Sequence[str]) -> list[str]:
        raise CacheMiss(f"{len(prompts)} prompts are not in the judge cache")


def call_counts(prompts: Sequence[str], cache: Path) -> dict[str, int]:
    """Calls a pass makes, distinct prompts among them, and how many of those are cached."""
    keys = {prompt_key(p) for p in prompts}
    cached = set()
    if cache.exists():
        cached = {json.loads(line)["key"] for line in cache.read_text().splitlines()}
    return {
        "calls": len(prompts),
        "unique": len(keys),
        "cached": len(keys & cached),
        "to_run": len(keys - cached),
    }


def experts_pass(
    chains: list[Chain], pairs: list[tuple[Chain, Pair]], judge: Judge
) -> tuple[list[dict], list[dict]]:
    """One-to-all P/R/F1 of every ordered expert pair (`paper` marks the paper's 450), and each
    expert's strict country guess read from the full round text, where the guess is.

    The recall prompts of (A, B) are the precision prompts of (B, A), so the two directions share
    every judge call: F1 is the same either way and P and R swap.
    """
    scores = score_pairs([(c.text, p.chain) for c, p in pairs], judge, VARIANT)
    pair_rows = [
        {
            "round": c.round.key,
            "author": c.author,
            "expert": p.expert,
            "paper": paper_expert_pair(c.author, p.expert),
            **asdict(s),
        }
        for (c, p), s in zip(pairs, scores, strict=True)
    ]
    shared = {c.round.key for c, _ in pairs}
    replies = judge.complete([build_country_prompt(c.full) for c in chains])
    chain_rows = []
    for chain, reply in zip(chains, replies, strict=True):
        guess = parse_country(reply)
        chain_rows.append(
            {
                "round": chain.round.key,
                "author": chain.author,
                "shared": int(chain.round.key in shared),
                "truth": chain.round.country,
                "guess": guess or "",
                "correct": int(guess == chain.round.country),
            }
        )
    return pair_rows, chain_rows


def pairwise_pass(pairs: list[tuple[Chain, Pair]], judge: Judge) -> list[dict]:
    """Repo-rubric score of every (expert point, candidate point), FIRST the expert point alone and
    SECOND the candidate point alone. scores[i][j]: expert point i vs candidate point j, or None."""
    points = [(extract_points(p.chain), extract_points(c.text)) for c, p in pairs]
    prompts = [build_prompt(g, x, VARIANT) for gt, cand in points for g in gt for x in cand]
    parsed = iter([parse_score(r) for r in judge.complete(prompts)])
    return [
        {
            "round": c.round.key,
            "author": c.author,
            "expert": p.expert,
            "expert_points": gt,
            "candidate_points": cand,
            "scores": [[next(parsed) for _ in cand] for _ in gt],
        }
        for (c, p), (gt, cand) in zip(pairs, points, strict=True)
    ]


def build_specificity_prompt(point: str) -> str:
    return load_template("specificity.txt").replace("{point}", point)


def parse_specificity(reply: str) -> float | None:
    """The "specificity" rating; None if missing or outside 0-100."""
    found = _SPECIFICITY.search(reply)
    if found is None:
        return None
    value = float(found.group(1))
    return value if 0 <= value <= 100 else None


def specificity_pass(chains: list[Chain], judge: Judge) -> dict[str, float | None]:
    """Specificity of each distinct point of the set's scored chains."""
    points = list(dict.fromkeys(p for c in chains for p in extract_points(c.text)))
    replies = judge.complete([build_specificity_prompt(p) for p in points])
    return {p: parse_specificity(r) for p, r in zip(points, replies, strict=True)}


def build_location_prompt(chain: str) -> str:
    return load_template("location.txt").replace("{chain}", chain)


def parse_location(reply: str) -> Place | None:
    """The first JSON object's place and coordinates; all null is a chain without a guess. None if
    there is no valid object, or the coordinates are not two in-range numbers or two nulls."""
    found = _JSON_OBJECT.search(reply)
    if found is None:
        return None
    try:
        obj = json.loads(found.group(0))
    except json.JSONDecodeError:
        return None
    name, lat, lon = obj.get("place"), obj.get("lat"), obj.get("lon")
    if name is not None and not isinstance(name, str):
        return None
    if lat is None and lon is None:
        return Place(name, None, None)
    numbers = all(isinstance(x, int | float) and not isinstance(x, bool) for x in (lat, lon))
    if not numbers or abs(lat) > 90 or abs(lon) > 180:
        return None
    return Place(name, float(lat), float(lon))


def location_pass(chains: list[Chain], judge: Judge) -> list[dict]:
    """The final place each chain guesses, read from its full text."""
    replies = judge.complete([build_location_prompt(c.full) for c in chains])
    rows = []
    for chain, reply in zip(chains, replies, strict=True):
        place = parse_location(reply)
        found = place or Place(None, None, None)
        rows.append(
            {
                "round": chain.round.key,
                "author": chain.author,
                "parsed": place is not None,
                "place": found.name,
                "lat": found.lat,
                "lon": found.lon,
            }
        )
    return rows


def grounded_request(round_key: str, points: Sequence[str]) -> str:
    """The round key on the first line, then the prompt, so the cache key covers the image."""
    statements = "\n".join(f"{i}. {p}" for i, p in enumerate(points, start=1))
    return f"{round_key}\n" + load_template("grounded.txt").replace("{statements}", statements)


def split_request(request: str) -> tuple[str, str]:
    key, prompt = request.split("\n", 1)
    return key, prompt


def parse_verdicts(reply: str, n: int) -> list[str | None]:
    """ "yes", "no" or "na" for statements 1..n from "N: verdict" lines; the first verdict for a
    number counts, and a statement without one is None."""
    verdicts: list[str | None] = [None] * n
    for number, word in _VERDICT.findall(reply):
        i = int(number) - 1
        if 0 <= i < n and verdicts[i] is None:
            verdicts[i] = word.lower().replace("/", "")
    return verdicts


def grounded_pass(chains: list[Chain], judge: Judge) -> list[dict]:
    """A vision judge's verdict on each point of each scored chain, against the round's image."""
    points = [extract_points(c.text) for c in chains]
    requests = [grounded_request(c.round.key, p) for c, p in zip(chains, points, strict=True) if p]
    replies = iter(judge.complete(requests))
    return [
        {
            "round": c.round.key,
            "author": c.author,
            "points": p,
            "verdicts": parse_verdicts(next(replies), len(p)) if p else [],
        }
        for c, p in zip(chains, points, strict=True)
    ]


def run_pass(pass_name: str, set_name: str, data: Path, out: Path, judge: Judge) -> Any:
    """Run one pass on one chain set; returns that pass function's result."""
    if pass_name == "experts" and set_name != EXPERTS:
        raise ValueError("the experts pass takes only the experts set")
    chains, pairs = load_set(set_name, data, out)
    if pass_name == "experts":
        return experts_pass(chains, pairs, judge)
    if pass_name == "pairwise":
        return pairwise_pass(pairs, judge)
    if pass_name == "specificity":
        return specificity_pass(chains, judge)
    if pass_name == "location":
        return location_pass(chains, judge)
    if pass_name == "grounded":
        return grounded_pass(chains, judge)
    raise ValueError(f"unknown pass {pass_name!r}")


def _mean(values: list[float]) -> float:
    return fmean(values) if values else float("nan")


def describe(pass_name: str, set_name: str, result: Any) -> str:
    """A few summary lines for the job log."""
    head = f"{pass_name} {set_name}"
    if pass_name == "experts":
        pair_rows, chain_rows = result
        lines = [head]
        for label, rows in [
            ("paper pairs (1&2, 2&3, 1&3)", [r for r in pair_rows if r["paper"]]),
            ("reverse pairs", [r for r in pair_rows if not r["paper"]]),
            ("all ordered pairs", pair_rows),
        ]:
            p, r, f = (_mean([row[k] for row in rows]) for k in ("precision", "recall", "f1"))
            unparsed = sum(row["n_unparsed"] for row in rows)
            lines.append(
                f"  {label}: n={len(rows)} P={p:.2f} R={r:.2f} F1={f:.2f} unparsed={unparsed}"
            )
        paper = PAPER_EXPERTS
        lines.append(
            f"  paper Table 2: n=450 P={paper['precision']} R={paper['recall']} F1={paper['f1']}"
            f" (95% CI {paper['f1_ci'][0]}-{paper['f1_ci'][1]})"
        )
        for expert in sorted({row["author"] for row in chain_rows}):
            rows = [row for row in chain_rows if row["author"] == expert]
            mine = [row["correct"] for row in rows]
            shared = [row["correct"] for row in rows if row["shared"]]
            lines.append(
                f"  {expert} country: shared rounds {100 * _mean(shared):.2f} (n={len(shared)},"
                f" paper {PAPER_EXPERT_COUNTRY.get(expert)}), all {100 * _mean(mine):.2f}"
                f" (n={len(mine)})"
            )
        return "\n".join(lines)
    if pass_name == "pairwise":
        cells = [s for row in result for line in row["scores"] for s in line]
        parsed = [s for s in cells if s is not None]
        return (
            f"{head}: pairs={len(result)} cells={len(cells)} mean={_mean(parsed):.2f}"
            f" unparsed={len(cells) - len(parsed)}"
        )
    if pass_name == "specificity":
        parsed = [s for s in result.values() if s is not None]
        return (
            f"{head}: points={len(result)} mean={_mean(parsed):.2f}"
            f" unparsed={len(result) - len(parsed)}"
        )
    if pass_name == "location":
        located = sum(row["lat"] is not None for row in result)
        no_guess = sum(row["parsed"] and row["place"] is None for row in result)
        unparsed = sum(not row["parsed"] for row in result)
        return (
            f"{head}: chains={len(result)} with_coordinates={located} no_guess={no_guess}"
            f" unparsed={unparsed}"
        )
    counts = Counter(v or "unparsed" for row in result for v in row["verdicts"])
    return f"{head}: chains={len(result)} " + " ".join(
        f"{k}={counts[k]}" for k in ("yes", "no", "na", "unparsed")
    )
