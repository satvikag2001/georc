"""Load GeoRC rounds, expert chains and released model chains from disk."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

RELEASED_TAGS: tuple[str, ...] = (
    "gemini25_flash",
    "gemini_3_pro",
    "gemini_pro",
    "gemma3_12b",
    "google_downsampled",
    "gpt4",
    "gpt5",
    "llama3_11b",
    "qwen",
    "qwen3",
    "hallucinated",
    "paraphrased",
)
_ROUND_HEADER = re.compile(r"\bRound (\d+)\b")

# Rounds with no `streakLocationCode`, filled from their coordinates (the paper's country
# accuracies are all multiples of 1/500, so every round had a ground truth).
COUNTRY_OVERRIDES: dict[tuple[str, int], str] = {
    ("8Uo6ejwXYqmp9av3", 4): "kr",  # lat 38.18, lng 128.60: Sokcho, South Korea
}


@dataclass(frozen=True)
class Round:
    challenge: str
    index: int  # 1-based
    country: str  # ISO 3166-1 alpha-2, lower case
    image: Path

    @property
    def key(self) -> str:
        return f"{self.challenge}_{self.index}"


@dataclass(frozen=True)
class Pair:
    round: Round
    expert: str  # e.g. "Human_Expert_2"
    chain: str


def parse_expert_rounds(text: str) -> dict[int, str]:
    """Split a Human_Expert_N.txt file into one chain per round number.

    As the paper's `parse_reasoning_chains`: lines containing "reasoning" are dropped and a round
    ends at its first line containing "conclusion". Rounds are keyed by their "Round N" header,
    not by position; on the released files the two agree.
    """
    pieces = _ROUND_HEADER.split(text)
    rounds: dict[int, str] = {}
    for number, body in zip(pieces[1::2], pieces[2::2], strict=True):
        kept = []
        for line in body.split("\n"):
            lowered = line.strip().lower()
            if "reasoning" in lowered:
                continue
            if "conclusion" in lowered:
                break
            kept.append(line.strip())
        rounds[int(number)] = "\n".join(kept)
    return rounds


def load_rounds(root: Path) -> list[Round]:
    """All rounds under `root`, ordered by challenge name then round index."""
    rounds = []
    for folder in sorted(root.iterdir()):
        meta = folder / f"{folder.name}_rounds_metadata.json"
        if not meta.exists():
            continue
        for i, entry in enumerate(json.loads(meta.read_text()), start=1):
            code = entry.get("streakLocationCode") or COUNTRY_OVERRIDES.get((folder.name, i))
            if not code:
                raise ValueError(f"{folder.name} round {i}: no country code and no override")
            rounds.append(Round(folder.name, i, code.lower(), folder / f"{folder.name}_{i}.png"))
    return rounds


def load_pairs(root: Path, rounds: list[Round]) -> list[Pair]:
    """One pair per (round, expert who wrote a chain for it), ordered by round then expert."""
    experts: dict[str, dict[str, dict[int, str]]] = {}
    pairs = []
    for rnd in rounds:
        if rnd.challenge not in experts:
            files = sorted((root / rnd.challenge).glob("Human_Expert_*.txt"))
            experts[rnd.challenge] = {f.stem: parse_expert_rounds(f.read_text()) for f in files}
        for expert, chains in experts[rnd.challenge].items():
            if rnd.index in chains:
                pairs.append(Pair(rnd, expert, chains[rnd.index]))
    return pairs


def chain_prefix(tag: str) -> str:
    """Released chains of `tag` are named <prefix><round index>.txt."""
    if tag in ("hallucinated", "paraphrased"):
        return f"{tag}_reasoning_chain_"
    return f"candidate_reasoning_chain_{tag}_"


def released_chain_path(root: Path, rnd: Round, tag: str) -> Path:
    return root / rnd.challenge / f"{chain_prefix(tag)}{rnd.index}.txt"
