"""Experimental conditions: which model, prompt and image handling produce a set of chains."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Condition:
    id: str
    model: str  # key in georc.models.GENERATORS
    prompt: str  # template file in georc/templates
    mm_kwargs: dict | None = None  # per-request processor options, e.g. Gemma pan-and-scan


CONDITIONS: dict[str, Condition] = {
    c.id: c
    for c in [
        # Paper baselines: Fig. 21 prompt, default image handling.
        Condition("G0", "gemma3-12b", "gen_paper.txt"),
        Condition("Q0", "qwen2.5-vl-7b", "gen_paper.txt"),
    ]
}
