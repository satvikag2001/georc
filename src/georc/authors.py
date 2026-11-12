"""Lay out and summarize runs of the GeoRC authors' released scorer (score.py at a7a1679)."""

import json
import math
import os
import re
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev

from georc.data import chain_prefix, load_rounds
from georc.judge import parse_score

TAGS: tuple[str, ...] = ("qwen", "gemma3_12b", "llama3_11b", "qwen3", "hallucinated")
# How the patched score.py reads a judge reply (GEORC_SCORE_KEY); "first_number" is our
# scorer's parse_score, for comparison only.
SCORE_KEYS: tuple[str, ...] = ("fallback", "precision", "released")
READINGS: tuple[str, ...] = (*SCORE_KEYS, "first_number")
ROUNDS = 5  # EvaluationPipeline.run(rounds=5)

# P, R, F1 in Table 2 of arXiv v1 (also the leaderboard.json at a7a1679). v2 (gates.TABLE2)
# changed every one; country accuracy did not change.
TABLE2_V1: dict[str, tuple[float, float, float]] = {
    "qwen": (29.06, 44.22, 31.63),
    "gemma3_12b": (38.20, 32.77, 31.21),
    "llama3_11b": (30.20, 30.72, 25.86),
    "qwen3": (35.54, 22.17, 23.81),
    "hallucinated": (33.50, 17.59, 18.48),
}

_JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)  # as in their _get_llm_score


@dataclass(frozen=True)
class Unit:
    """One chain set, one expert, one challenge: five scored pairs."""

    tag: str
    expert: str  # e.g. "Human_Expert_2"
    challenge: str

    def path(self, run_dir: Path) -> Path:
        return run_dir / "units" / self.tag / self.expert / f"{self.challenge}.json"


def list_units(data: Path, tags: Iterable[str]) -> list[Unit]:
    """Every unit, by tag, then challenge, then expert."""
    challenges = sorted({r.challenge for r in load_rounds(data)})
    experts = {c: sorted(f.stem for f in (data / c).glob("Human_Expert_*.txt")) for c in challenges}
    return [
        Unit(tag, expert, challenge)
        for tag in tags
        for challenge, names in experts.items()
        for expert in names
    ]


def shard(units: Sequence[Unit], spec: str) -> list[Unit]:
    """Shard "i/n" (1-based) of the units, taking every n-th; "1/1" returns all of them."""
    index, count = (int(x) for x in spec.split("/"))
    if not 1 <= index <= count:
        raise ValueError(f"shard must be i/n with 1 <= i <= n, got {spec}")
    return list(units[index - 1 :: count])


def build_view(data: Path, views: Path, unit: Unit) -> Path:
    """A folder laid out as their pipeline expects, for this unit only.

    Their code reads the expert chain from <challenge>/Human_Chain_3.txt, so this links the unit's
    expert file under that name, next to the five candidate files; scoring every expert this way
    gives the paper's 800 pairs per chain set. Returns <views>/<tag>/<expert>/<challenge>, whose
    only subfolder is <challenge>. A missing file is an error here; their pipeline skips it.
    """
    source = data / unit.challenge
    root = views / unit.tag / unit.expert / unit.challenge
    folder = root / unit.challenge
    folder.mkdir(parents=True, exist_ok=True)
    prefix = chain_prefix(unit.tag)
    links = {"Human_Chain_3.txt": source / f"{unit.expert}.txt"}
    links |= {f"{prefix}{i}.txt": source / f"{prefix}{i}.txt" for i in range(1, ROUNDS + 1)}
    for name, target in links.items():
        if not target.exists():
            raise FileNotFoundError(target)
        link = folder / name
        if link.is_symlink() and link.readlink() == target.resolve():
            continue
        link.unlink(missing_ok=True)
        link.symlink_to(target.resolve())
    return root


def unit_seed(unit: Unit, base: int) -> int:
    """A fixed sampling seed per unit, so a resubmitted unit draws the same replies."""
    return (base + zlib.crc32(f"{unit.tag}/{unit.expert}/{unit.challenge}".encode())) % 2**31


def read_score(reply: str, metric: str, reading: str) -> float:
    """One judge reply as a 0-100 score, read the way `reading` says.

    The SCORE_KEYS readings repeat the patched `_get_llm_score`: the first {...} is parsed as JSON
    and a key is read; anything unparseable scores 0.
    """
    if reading not in READINGS:
        raise ValueError(f"unknown reading: {reading}")
    if reading == "first_number":
        value = parse_score(reply)
        return 0.0 if value is None else value
    match = _JSON_OBJECT.search(reply)
    if match is None:
        return 0.0
    try:
        data = json.loads(match.group(0))
        if reading == "released":
            return float(data.get(metric, 0))
        if reading == "precision":
            return float(data.get("precision", 0))
        return float(data.get(metric, data.get("precision", 0)))
    except (ValueError, KeyError, TypeError):
        return 0.0


def pair_metrics(calls: Sequence[dict], reading: str) -> dict[str, float]:
    """Precision, recall and F1 of one pair from its judge calls, as BipartiteLLMScorer.score."""
    scores = {"precision": [], "recall": []}
    for call in calls:
        scores[call["metric"]].append(read_score(call["reply"], call["metric"], reading))
    precision, recall = scores["precision"], scores["recall"]
    if not precision or not recall:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    p, r = sum(precision) / len(precision), sum(recall) / len(recall)
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r > 0 else 0.0}


def summarize(pairs: Sequence[dict]) -> dict[str, float]:
    """Means of per-pair values (as our scorer), plus F1 of the means and a normal-theory CI.

    Neither their code nor the paper says how Table 6's 95% CI was computed, so `f1_ci95`
    (mean +/- 1.96 standard errors over pairs) is a reference only.
    """
    p, r, f1 = (fmean(x[k] for x in pairs) for k in ("precision", "recall", "f1"))
    half = 1.96 * stdev(x["f1"] for x in pairs) / math.sqrt(len(pairs)) if len(pairs) > 1 else 0
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "f1_of_means": 2 * p * r / (p + r) if p + r > 0 else 0.0,
        "f1_ci95": [f1 - half, f1 + half],
        "n": len(pairs),
    }


def completed_units(run_dir: Path, tag: str) -> list[Unit]:
    """Units of `tag` already written under `run_dir` (no dataset needed)."""
    return [
        Unit(tag, path.parent.name, path.stem)
        for path in sorted((run_dir / "units" / tag).glob("*/*.json"))
    ]


def load_pairs(run_dir: Path, units: Sequence[Unit]) -> list[dict]:
    """Per-pair rows: their result record plus tag, expert, score key and the judge calls."""
    rows = []
    for unit in units:
        record = json.loads(unit.path(run_dir).read_text())
        meta = {"tag": unit.tag, "expert": unit.expert, "score_key": record["score_key"]}
        for result, calls in zip(record["results"], record["calls"], strict=True):
            rows.append({**meta, **result, "calls": calls})
    return rows


def write_json(path: Path, obj: object) -> None:
    """Write via a temporary file and rename, so a killed job never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)
