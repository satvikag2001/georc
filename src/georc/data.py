"""Load GeoRC challenges, rounds, expert chains and released model chains from disk."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

# Chain sets released in the HF dataset, by file-name tag.
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
_BASELINE_TAGS = {"hallucinated", "paraphrased"}
_ROUND_HEADER = re.compile(r"\bRound (\d+)\b")

# Rounds whose metadata has no `streakLocationCode`. The paper's country accuracies are all
# multiples of 1/500, so every round had a ground truth; we fill the gap from the coordinates.
COUNTRY_OVERRIDES: dict[tuple[str, int], str] = {
    ("8Uo6ejwXYqmp9av3", 4): "kr",  # lat 38.18, lng 128.60: Sokcho, South Korea
}


@dataclass(frozen=True)
class Round:
    """One GeoGuessr location: a challenge folder plus a 1-based round index."""

    challenge: str
    index: int
    country: str  # ISO 3166-1 alpha-2, lower case (metadata `streakLocationCode`)
    image: Path

    @property
    def key(self) -> str:
        return f"{self.challenge}_{self.index}"


@dataclass(frozen=True)
class Pair:
    """A round scored against one expert's chain for that round."""

    round: Round
    expert: str  # e.g. "Human_Expert_2"
    chain: str


def parse_expert_rounds(text: str) -> dict[int, str]:
    """Split a Human_Expert_N.txt file into one chain per round.

    Follows the paper's `parse_reasoning_chains`: within a round, lines containing "reasoning"
    are dropped and reading stops at the first line containing "conclusion". Rounds are keyed by
    the number in their "Round N" header rather than by position; on the released files (no text
    before "Round 1", rounds in order) the two are identical.
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


def _challenge_dirs(root: Path) -> list[Path]:
    return sorted(
        d for d in root.iterdir() if d.is_dir() and (d / f"{d.name}_rounds_metadata.json").exists()
    )


def load_rounds(root: Path) -> list[Round]:
    """All rounds under `root`, ordered by challenge name then round index."""
    rounds = []
    for folder in _challenge_dirs(root):
        meta = json.loads((folder / f"{folder.name}_rounds_metadata.json").read_text())
        for i, entry in enumerate(meta, start=1):
            code = entry.get("streakLocationCode") or COUNTRY_OVERRIDES.get((folder.name, i))
            if not code:
                raise ValueError(f"{folder.name} round {i}: no country code and no override")
            rounds.append(
                Round(
                    challenge=folder.name,
                    index=i,
                    country=code.lower(),
                    image=folder / f"{folder.name}_{i}.png",
                )
            )
    return rounds


def load_pairs(root: Path, rounds: list[Round]) -> list[Pair]:
    """One pair per (round, expert who wrote a chain for it), ordered by round then expert."""
    parsed: dict[tuple[str, str], dict[int, str]] = {}
    experts_by_challenge: dict[str, list[str]] = {}
    for rnd in rounds:
        if rnd.challenge in experts_by_challenge:
            continue
        files = sorted((root / rnd.challenge).glob("Human_Expert_*.txt"))
        experts_by_challenge[rnd.challenge] = [f.stem for f in files]
        for f in files:
            parsed[(rnd.challenge, f.stem)] = parse_expert_rounds(f.read_text())
    pairs = []
    for rnd in rounds:
        for expert in experts_by_challenge[rnd.challenge]:
            chain = parsed[(rnd.challenge, expert)].get(rnd.index)
            if chain is not None:
                pairs.append(Pair(round=rnd, expert=expert, chain=chain))
    return pairs


def released_chain_path(root: Path, rnd: Round, tag: str) -> Path:
    """Path of a released chain for `rnd`; `tag` is one of RELEASED_TAGS."""
    if tag in _BASELINE_TAGS:
        name = f"{tag}_reasoning_chain_{rnd.index}.txt"
    else:
        name = f"candidate_reasoning_chain_{tag}_{rnd.index}.txt"
    return root / rnd.challenge / name
