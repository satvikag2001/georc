"""Print the vLLM API details the generation and judge code depends on (run on PACE)."""

import inspect

import vllm
from vllm import LLM, SamplingParams

print("vllm", vllm.__version__)
print("LLM.__init__", inspect.signature(LLM.__init__))
print("LLM.chat", inspect.signature(LLM.chat))
print("LLM.generate", inspect.signature(LLM.generate))
print("has get_default_sampling_params:", hasattr(LLM, "get_default_sampling_params"))
print("SamplingParams", inspect.signature(SamplingParams))
