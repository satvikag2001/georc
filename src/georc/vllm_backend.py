"""vLLM-backed judge (PACE only; vllm is not installed locally)."""

from collections.abc import Sequence

from vllm import LLM

from georc.models import JUDGE


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
