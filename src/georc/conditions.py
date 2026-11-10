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
        # Baseline with the prompt from the authors' released code (generate.py main_prompt).
        Condition("G0R", "gemma3-12b", "gen_repo.txt"),
        # Fix A: Gemma pan-and-scan, so high-resolution panoramas are not squashed to one crop.
        Condition("A1", "gemma3-12b", "gen_paper.txt", {"do_pan_and_scan": True}),
    ]
}
