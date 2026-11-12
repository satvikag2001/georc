import pytest

from georc.country import (
    build_country_match_prompt,
    build_country_prompt,
    country_accuracy,
    parse_country,
    parse_match,
)
from georc.judge import CachedJudge, FunctionJudge, build_prompt, parse_score
from georc.score import score_pairs, summarize


def split_prompt(prompt: str) -> tuple[str, str]:
    """Recover (first chain, second point) from a judge prompt built by build_prompt."""
    head, second = prompt.rsplit("SECOND REASONING CHAIN:", 1)
    first = head.rsplit("FIRST REASONING CHAIN:", 1)[1]
    return first.strip(), second.strip()


def exact_match_judge() -> FunctionJudge:
    """Scores 100 when the point is one of the first chain's points verbatim, else 0."""

    def reply(prompt: str) -> str:
        first, second = split_prompt(prompt)
        return '{"precision": 100}' if second in first.split(" || ") else '{"precision": 0}'

    return FunctionJudge(reply)


def test_parse_score_reads_the_number_whatever_the_key():
    # The released code looked up "recall" in replies that use "precision", scoring recall 0.
    assert parse_score('{"recall": 40}') == 40.0


def test_parse_score_ignores_text_around_the_json():
    assert parse_score('Sure! {"precision": 72.5} Hope that helps.') == 72.5


def test_parse_score_returns_none_without_a_json_object():
    assert parse_score("precision 80") is None


def test_parse_score_returns_none_for_json_without_a_number():
    assert parse_score('{"precision": "high"}') is None


def test_parse_score_clips_to_the_0_to_100_range():
    assert parse_score('{"precision": 150}') == 100.0
    assert parse_score('{"precision": -5}') == 0.0


def test_paper_prompt_puts_both_chains_after_their_markers():
    prompt = build_prompt("Cyrillic signage || White tree trunks", "Cyrillic on a sign")
    assert split_prompt(prompt) == ("Cyrillic signage || White tree trunks", "Cyrillic on a sign")
    assert "SAME EVIDENCE TYPE" in prompt  # Fig. 14 rubric, not the released-code rubric


def test_prompt_survives_braces_in_chain_text():
    prompt = build_prompt("Sign reads {EXIT}", "a {placeholder} point")
    assert split_prompt(prompt) == ("Sign reads {EXIT}", "a {placeholder} point")


def test_repo_prompt_variant_is_the_released_rubric():
    prompt = build_prompt("A || B", "A", variant="repo")
    assert "Only award points if there is logical evidence" in prompt
    assert '{"precision": <score>}' in prompt


def test_one_to_all_scores_on_a_hand_computed_pair():
    candidate = "1. Cyrillic signage\n2. The sky is blue\n3. The road is paved"
    expert = "Cyrillic signage\nWhite tree trunks"
    [score] = score_pairs([(candidate, expert)], exact_match_judge())
    assert score.precision == pytest.approx(100 / 3)  # 1 of 3 candidate points supported
    assert score.recall == pytest.approx(50.0)  # 1 of 2 expert points found
    assert score.f1 == pytest.approx(40.0)
    assert (score.n_candidate_points, score.n_expert_points) == (3, 2)


def test_empty_candidate_scores_zero_without_calling_the_judge():
    def never(prompt: str) -> str:
        raise AssertionError("judge should not be called")

    [score] = score_pairs([("", "Cyrillic signage")], FunctionJudge(never))
    assert (score.precision, score.recall, score.f1) == (0.0, 0.0, 0.0)


def test_unparseable_replies_score_zero_and_are_counted():
    [score] = score_pairs([("Cyrillic signage", "Cyrillic signage")], FunctionJudge(lambda p: "?"))
    assert score.f1 == 0.0
    assert score.n_unparsed == 2  # one precision call, one recall call


def test_all_judge_calls_for_all_pairs_go_out_in_one_batch():
    batches = []

    class RecordingJudge:
        def complete(self, prompts):
            batches.append(len(prompts))
            return ['{"precision": 50}'] * len(prompts)

    pairs = [("Cyrillic signage\nPainted trunks", "Cyrillic signage")] * 3
    score_pairs(pairs, RecordingJudge())
    assert batches == [9]  # 3 pairs x (2 precision + 1 recall)


def test_summary_f1_is_the_mean_of_per_pair_f1_not_f1_of_means():
    perfect = ("Cyrillic signage", "Cyrillic signage")
    lopsided = ("Cyrillic signage\nThe sky is blue", "Cyrillic signage")  # P=50, R=100, F1=66.7
    summary = summarize(score_pairs([perfect, lopsided], exact_match_judge()))
    assert summary["precision"] == pytest.approx(75.0)
    assert summary["recall"] == pytest.approx(100.0)
    assert summary["f1"] == pytest.approx((100 + 200 / 3) / 2)
    assert summary["n"] == 2


def test_cached_judge_answers_repeat_prompts_from_disk(tmp_path):
    cache = tmp_path / "judge_cache.jsonl"
    first = CachedJudge(FunctionJudge(lambda p: '{"precision": 42}'), cache)
    assert first.complete(["p1", "p2"]) == ['{"precision": 42}'] * 2

    def never(prompt: str) -> str:
        raise AssertionError("cached prompt was sent to the judge again")

    second = CachedJudge(FunctionJudge(never), cache)
    assert second.complete(["p2", "p1"]) == ['{"precision": 42}'] * 2


def test_cached_judge_saves_finished_chunks_before_a_failure(tmp_path):
    cache = tmp_path / "judge_cache.jsonl"
    calls = []

    def flaky(prompt: str) -> str:
        calls.append(prompt)
        if prompt == "p3":
            raise RuntimeError("preempted")
        return '{"precision": 1}'

    with pytest.raises(RuntimeError):
        CachedJudge(FunctionJudge(flaky), cache, chunk_size=2).complete(["p1", "p2", "p3"])
    resumed = CachedJudge(FunctionJudge(lambda p: '{"precision": 9}'), cache, chunk_size=2)
    assert resumed.complete(["p1", "p2", "p3"]) == [
        '{"precision": 1}',
        '{"precision": 1}',
        '{"precision": 9}',
    ]


def test_country_prompt_contains_the_chain():
    assert build_country_prompt("Conclusion: Kyrgyzstan").endswith("Conclusion: Kyrgyzstan")


def test_parse_country_returns_lower_case_iso_code():
    assert parse_country('{"country": "KG"}') == "kg"


@pytest.mark.parametrize("reply", ['{"country": "XX"}', "Kyrgyzstan", '{"country": "KGZ"}'])
def test_parse_country_returns_none_when_there_is_no_two_letter_guess(reply):
    assert parse_country(reply) is None


def test_country_accuracy_is_a_percentage_and_missing_guesses_count_as_wrong():
    assert country_accuracy(["kg", None, "rs", "br"], ["kg", "my", "rs", "ar"]) == 50.0


def test_match_prompt_gives_the_judge_the_iso_code_and_the_chain():
    prompt = build_country_match_prompt("Conclusion: Kazakhstan or Kyrgyzstan", "kg")
    assert "TRUE COUNTRY: KG" in prompt  # the paper compares against ISO codes
    assert prompt.endswith("Conclusion: Kazakhstan or Kyrgyzstan")


@pytest.mark.parametrize(
    ("reply", "expected"),
    [('{"match": true}', True), ('{"match": false}', False), ('ok {"match": True}', True)],
)
def test_parse_match_reads_the_boolean(reply, expected):
    assert parse_match(reply) is expected


@pytest.mark.parametrize("reply", ['{"match": "yes"}', "true", "{}"])
def test_parse_match_returns_none_for_anything_else(reply):
    assert parse_match(reply) is None
