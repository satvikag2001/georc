"""vLLM-backed judge (PACE only; vllm is not installed locally)."""

from collections.abc import Sequence

from vllm import LLM

from georc.models import JUDGE, ModelSpec


class VLLMJudge:
    """Qwen3-4B judge. Like the paper's `text_to_text`, each prompt is wrapped as one user
    message with the model's chat template, and sampling uses the model's own defaults unless
    `greedy` is set (the G1 fallback). A fixed per-request seed makes a given prompt's reply
    reproducible, which is what the on-disk cache assumes."""

    def __init__(
        self, spec: ModelSpec = JUDGE, seed: int = 0, greedy: bool = False, max_model_len=32768
    ) -> None:
        self.llm = LLM(
            model=spec.repo,
            revision=spec.revision,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=0.85,
            enable_prefix_caching=True,
            seed=seed,
        )
        params = self.llm.get_default_sampling_params()
        params.max_tokens = spec.max_tokens
        params.seed = seed
        if greedy:
            params.temperature = 0.0
        self.params = params

    def complete(self, prompts: Sequence[str]) -> list[str]:
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        outputs = self.llm.chat(conversations, self.params, use_tqdm=True)
        return [o.outputs[0].text for o in outputs]
