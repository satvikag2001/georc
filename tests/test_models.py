import re

from georc.models import GENERATORS, JUDGE


def test_every_model_is_pinned_to_a_full_commit_hash():
    for spec in [*GENERATORS.values(), JUDGE]:
        assert re.fullmatch(r"[0-9a-f]{40}", spec.revision), spec.repo


def test_generator_settings_match_the_paper_appendix():
    gemma, qwen = GENERATORS["gemma3-12b"], GENERATORS["qwen2.5-vl-7b"]
    assert (gemma.temperature, gemma.max_tokens) == (1.0, 512)
    assert (qwen.temperature, qwen.max_tokens) == (0.7, 32768)


def test_judge_keeps_its_own_sampling_defaults():
    assert JUDGE.repo == "Qwen/Qwen3-4B-Instruct-2507"
    assert JUDGE.temperature is None
    assert JUDGE.max_tokens == 2048
