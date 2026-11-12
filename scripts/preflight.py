"""GPU smoke test for one model, run offline in pace/preflight.sbatch; non-zero exit on failure."""

import argparse
import time
from pathlib import Path

from PIL import Image
from vllm import LLM

from georc.country import build_country_prompt, parse_country
from georc.data import load_rounds
from georc.generate import build_messages
from georc.judge import build_prompt, parse_score
from georc.models import GENERATORS


def run_judge() -> None:
    from georc.vllm_backend import VLLMJudge

    judge = VLLMJudge(seed=0)
    print("sampling:", judge.params)
    first = "Cyrillic writing on the gas station sign || White painted bottoms of the trees"
    replies = judge.complete(
        [
            build_prompt(first, "There is Cyrillic script on the sign at the gas station"),
            build_prompt(first, "The sky is blue with some clouds"),
            build_country_prompt("Cyrillic sign, poplar trees.\nConclusion: Kyrgyzstan, Bishkek"),
        ]
    )
    print("match reply:", replies[0], "->", parse_score(replies[0]))
    print("mismatch reply:", replies[1], "->", parse_score(replies[1]))
    print("country reply:", replies[2], "->", parse_country(replies[2]))


def one_chain(llm: LLM, params, image: Image.Image, mm_kwargs: dict | None = None) -> int:
    extra = {"mm_processor_kwargs": mm_kwargs} if mm_kwargs else {}
    start = time.time()
    [out] = llm.chat(build_messages("gen_paper.txt", image), params, **extra)
    seconds = time.time() - start
    completion = out.outputs[0]
    print(
        f"prompt_tokens={len(out.prompt_token_ids)} output_tokens={len(completion.token_ids)} "
        f"finish={completion.finish_reason} seconds={seconds:.1f}"
    )
    print(completion.text[:600])
    return len(out.prompt_token_ids)


def run_generator(name: str, data: Path) -> None:
    spec = GENERATORS[name]
    image = Image.open(load_rounds(data)[0].image).convert("RGB")
    llm = LLM(
        model=spec.repo,
        revision=spec.revision,
        dtype="bfloat16",
        max_model_len=40960 if name == "qwen2.5-vl-7b" else 8192,
        gpu_memory_utilization=0.85,
        seed=0,
        limit_mm_per_prompt={"image": 9},
    )
    params = llm.get_default_sampling_params()
    params.max_tokens = min(spec.max_tokens, 1024)
    if spec.temperature is not None:
        params.temperature = spec.temperature
    params.seed = 0
    print("sampling:", params)
    base = one_chain(llm, params, image)
    if name == "gemma3-12b":
        panned = one_chain(llm, params, image, {"do_pan_and_scan": True})
        print(f"pan-and-scan adds {panned - base} prompt tokens (expect several hundred)")
        if panned <= base:
            raise SystemExit("pan-and-scan did not change the input; use the A2 crop fallback")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", choices=["judge", "qwen-vl", "gemma"], required=True)
    args = parser.parse_args()
    if args.model == "judge":
        run_judge()
    else:
        run_generator({"qwen-vl": "qwen2.5-vl-7b", "gemma": "gemma3-12b"}[args.model], args.data)
    print("PREFLIGHT_OK", args.model)


if __name__ == "__main__":
    main()
