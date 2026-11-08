"""GPU smoke test for one model (run inside pace/preflight.sbatch, offline).

judge:   one-to-all calls on a matching and a mismatched point, plus one country call.
qwen-vl: one chain for the first round with the paper prompt.
gemma:   one chain at the default 896x896 input and one with pan-and-scan, comparing prompt
         token counts (pan-and-scan should add image crops).
Exits non-zero if a model fails to load or generate.
"""

import argparse
import time
from pathlib import Path

from PIL import Image
from vllm import LLM

from georc.country import build_country_prompt, parse_country
from georc.data import load_rounds
from georc.judge import build_prompt, load_template, parse_score
from georc.models import GENERATORS, JUDGE, ModelSpec


def make_llm(spec: ModelSpec, max_model_len: int, images: int) -> LLM:
    kwargs = dict(
        model=spec.repo,
        revision=spec.revision,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.85,
        seed=0,
    )
    if images:
        kwargs["limit_mm_per_prompt"] = {"image": images}
    return LLM(**kwargs)


def sampling(llm: LLM, spec: ModelSpec, max_tokens: int):
    params = llm.get_default_sampling_params()
    params.max_tokens = max_tokens
    if spec.temperature is not None:
        params.temperature = spec.temperature
    params.seed = 0
    print("sampling:", params)
    return params


def run_judge() -> None:
    llm = make_llm(JUDGE, max_model_len=8192, images=0)
    params = sampling(llm, JUDGE, max_tokens=64)
    first = "Cyrillic writing on the gas station sign || White painted bottoms of the trees"
    prompts = [
        build_prompt(first, "There is Cyrillic script on the sign at the gas station"),
        build_prompt(first, "The sky is blue with some clouds"),
        build_country_prompt("Cyrillic sign, poplar trees.\nConclusion: Kyrgyzstan, near Bishkek"),
    ]
    outs = llm.generate(prompts, params)
    replies = [o.outputs[0].text for o in outs]
    print("match reply:", replies[0], "->", parse_score(replies[0]))
    print("mismatch reply:", replies[1], "->", parse_score(replies[1]))
    print("country reply:", replies[2], "->", parse_country(replies[2]))


def one_chain(llm: LLM, params, image: Image.Image, mm_kwargs: dict | None = None):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": image},
                {"type": "text", "text": load_template("gen_paper.txt")},
            ],
        }
    ]
    extra = {"mm_processor_kwargs": mm_kwargs} if mm_kwargs else {}
    start = time.time()
    [out] = llm.chat(messages, params, **extra)
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
    llm = make_llm(spec, max_model_len=40960 if name == "qwen2.5-vl-7b" else 8192, images=9)
    params = sampling(llm, spec, max_tokens=min(spec.max_tokens, 1024))
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
