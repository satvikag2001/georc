"""vLLM-backed judges (PACE only; vllm is not installed locally)."""

from collections.abc import Mapping, Sequence
from pathlib import Path

from PIL import Image
from vllm import LLM, SamplingParams

from georc.extras import split_request
from georc.generate import image_messages
from georc.models import GENERATORS, JUDGE


class VLLMJudge:
    """Like the paper's `text_to_text`: each prompt is one user message in the chat template, with
    the model's own sampling unless `greedy` (G1 fallback). The fixed per-request seed makes a
    prompt's reply reproducible, which the on-disk cache assumes."""

    def __init__(self, seed: int = 0, greedy: bool = False) -> None:
        self.llm = LLM(
            model=JUDGE.repo,
            revision=JUDGE.revision,
            dtype="bfloat16",
            max_model_len=32768,
            gpu_memory_utilization=0.85,
            enable_prefix_caching=True,
            seed=seed,
        )
        params = self.llm.get_default_sampling_params()
        params.max_tokens = JUDGE.max_tokens
        params.seed = seed
        if greedy:
            params.temperature = 0.0
        self.params = params

    def complete(self, prompts: Sequence[str]) -> list[str]:
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        outputs = self.llm.chat(conversations, self.params, use_tqdm=True)
        return [o.outputs[0].text for o in outputs]


class VLLMVisionJudge:
    """Qwen2.5-VL-7B, greedy, one image per request. Requests are extras.grounded_request strings:
    the first line is the round key, which names the image; the rest is the prompt text."""

    def __init__(self, images: Mapping[str, Path], max_tokens: int) -> None:
        spec = GENERATORS["qwen2.5-vl-7b"]
        self.images = images
        self.llm = LLM(
            model=spec.repo,
            revision=spec.revision,
            dtype="bfloat16",
            max_model_len=40960,  # as in the PACE preflight that passed
            gpu_memory_utilization=0.85,
            limit_mm_per_prompt={"image": 1},
            seed=0,
        )
        # Plain greedy, without the model's default repetition penalty (1.05), which would push
        # against repeating the same verdict line after line.
        self.params = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=0)

    def complete(self, requests: Sequence[str]) -> list[str]:
        conversations = []
        for request in requests:
            key, text = split_request(request)
            image = Image.open(self.images[key]).convert("RGB")
            conversations.append(image_messages(text, image))
        outputs = self.llm.chat(conversations, self.params, use_tqdm=True)
        return [o.outputs[0].text for o in outputs]
