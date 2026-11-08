"""Pinned models and the decoding settings the paper used for each (GeoRC Appendix A.5.1).

Sampling starts from each model's own generation_config.json (what the paper's
`model.generate()` calls used) and overrides only what the paper states explicitly.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    revision: str  # HF commit, pinned 2026-11-08
    max_tokens: int
    temperature: float | None  # None: keep the generation_config value


GENERATORS: dict[str, ModelSpec] = {
    # Paper: temperature 1.0 and 512 max new tokens for Gemma.
    "gemma3-12b": ModelSpec(
        repo="google/gemma-3-12b-it",
        revision="96b6f1eccf38110c56df3a15bffe176da04bfd80",
        max_tokens=512,
        temperature=1.0,
    ),
    # Paper: temperature 0.7, max new tokens raised to 32768 so chains finish. The model's own
    # config is near-greedy (1e-6) with repetition_penalty 1.05; we keep the penalty and use the
    # paper's 0.7. Near-greedy is the documented fallback if gate G2 fails.
    "qwen2.5-vl-7b": ModelSpec(
        repo="Qwen/Qwen2.5-VL-7B-Instruct",
        revision="cc594898137f460bfe9f0759e9844b3ce807cfb5",
        max_tokens=32768,
        temperature=0.7,
    ),
}

# Paper: Qwen3-4B-Instruct-2507 judge, 2048 max new tokens. Its config samples
# (temperature 0.7, top_p 0.8, top_k 20); the released code used those defaults.
JUDGE = ModelSpec(
    repo="Qwen/Qwen3-4B-Instruct-2507",
    revision="cdbee75f17c01a7cc42f958dc650907174af0554",
    max_tokens=2048,
    temperature=None,
)
