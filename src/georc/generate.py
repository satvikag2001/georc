"""Generate reasoning chains for a condition with vLLM (PACE only), resumably.

Each chain is written as <out>/chains/<condition>/seed<k>/<round key>.txt with a .json beside it
holding token counts, finish reason and batch timing. A chain file appears only once complete
(write to .tmp, then rename), so a job preempted mid-batch is simply resubmitted.
"""

import json
import os
import time
from pathlib import Path

from PIL import Image

from georc.conditions import Condition
from georc.data import Pair, Round
from georc.judge import load_template

PILOT_CHALLENGES = 20


def chain_path(out: Path, condition_id: str, seed: int, rnd: Round) -> Path:
    return Path(out) / "chains" / condition_id / f"seed{seed}" / f"{rnd.key}.txt"


def pilot_rounds(rounds: list[Round]) -> list[Round]:
    """Rounds of the first PILOT_CHALLENGES challenges in name order (100 images)."""
    names = sorted({r.challenge for r in rounds})[:PILOT_CHALLENGES]
    return [r for r in rounds if r.challenge in names]


def restrict_pairs(pairs: list[Pair], rounds: list[Round]) -> list[Pair]:
    keys = {r.key for r in rounds}
    return [p for p in pairs if p.round.key in keys]


def pending_rounds(rounds: list[Round], out: Path, condition_id: str, seed: int) -> list[Round]:
    return [r for r in rounds if not chain_path(out, condition_id, seed, r).exists()]


def write_chain(path: Path, text: str, meta: dict) -> None:
    """Write the metadata, then the chain; the chain's rename is the commit point."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(meta))
    tmp = path.with_suffix(".txt.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def build_messages(prompt_template: str, image: Image.Image) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": image},
                {"type": "text", "text": load_template(prompt_template)},
            ],
        }
    ]


def generate(
    condition: Condition, rounds: list[Round], out: Path, seed: int, batch_size: int = 32
) -> int:
    """Generate every missing chain for one seed; returns how many were written."""
    from vllm import LLM  # import late: vllm exists only on PACE

    from georc.models import GENERATORS

    todo = pending_rounds(rounds, out, condition.id, seed)
    if not todo:
        return 0
    spec = GENERATORS[condition.model]
    llm = LLM(
        model=spec.repo,
        revision=spec.revision,
        dtype="bfloat16",
        max_model_len=spec.max_tokens + 8192,
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 9},
        seed=seed,
    )
    params = llm.get_default_sampling_params()
    params.max_tokens = spec.max_tokens
    if spec.temperature is not None:
        params.temperature = spec.temperature
    params.seed = seed
    extra = {"mm_processor_kwargs": condition.mm_kwargs} if condition.mm_kwargs else {}

    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        conversations = [
            build_messages(condition.prompt, Image.open(r.image).convert("RGB")) for r in batch
        ]
        began = time.time()
        outputs = llm.chat(conversations, params, use_tqdm=False, **extra)
        seconds = time.time() - began
        for rnd, output in zip(batch, outputs, strict=True):
            completion = output.outputs[0]
            meta = {
                "condition": condition.id,
                "model": spec.repo,
                "revision": spec.revision,
                "seed": seed,
                "prompt_tokens": len(output.prompt_token_ids),
                "output_tokens": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
                "batch_seconds": seconds,
                "batch_size": len(batch),
            }
            write_chain(chain_path(out, condition.id, seed, rnd), completion.text, meta)
        print(f"{condition.id} seed{seed}: {start + len(batch)}/{len(todo)} written", flush=True)
    return len(todo)
