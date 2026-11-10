import pytest

from georc.gates import OPEN_MODELS, TABLE2, check_country, check_g1, validation_rows


def paper_like_summaries() -> dict[str, dict[str, float]]:
    """Summaries equal to the paper's Table 2: every G1 check should pass."""
    out = {}
    for name, row in TABLE2.items():
        out[name] = {
            "precision": row.precision,
            "recall": row.recall,
            "f1": row.f1,
            "country_acc": row.country if row.country is not None else 0.0,
            "country_acc_match": row.country if row.country is not None else 0.0,
        }
    return out


def failed(checks) -> list[str]:
    return [c.name for c in checks if not c.passed]


def test_summaries_matching_the_paper_pass_every_g1_check():
    assert failed(check_g1(paper_like_summaries())) == []


def test_f1_one_point_outside_the_paper_ci_still_passes_and_beyond_fails():
    summaries = paper_like_summaries()
    summaries["qwen"]["f1"] = TABLE2["qwen"].f1_ci[1] + 1.0  # 31.15: the edge of the allowance
    assert failed(check_g1(summaries)) == []
    summaries["qwen"]["f1"] = TABLE2["qwen"].f1_ci[1] + 1.01
    assert "qwen F1 within paper CI +/- 1.0" in failed(check_g1(summaries))


def test_a_swapped_ranking_fails_even_when_each_f1_is_in_range():
    summaries = paper_like_summaries()
    summaries["llama3_11b"]["f1"] = 24.9  # in its range, but now above Gemma's 24.5
    summaries["gemma3_12b"]["f1"] = 24.5
    assert failed(check_g1(summaries)) == ["F1 ranking matches the paper"]


def test_country_accuracy_exactly_three_points_off_passes():
    summaries = paper_like_summaries()
    summaries["gemma3_12b"]["country_acc"] = 64.0 - 3.0
    assert failed(check_g1(summaries)) == []
    summaries["gemma3_12b"]["country_acc"] = 64.0 - 3.2
    assert failed(check_g1(summaries)) == ["gemma3_12b country accuracy within 3 points"]


def test_hallucinated_baseline_must_stay_within_two_points():
    summaries = paper_like_summaries()
    summaries["hallucinated"]["f1"] = 18.13 + 2.5
    assert failed(check_g1(summaries)) == ["hallucinated F1 within 2 points of 18.13"]


def test_missing_chain_set_is_an_error_not_a_silent_pass():
    summaries = paper_like_summaries()
    del summaries["qwen3"]
    with pytest.raises(KeyError):
        check_g1(summaries)


def test_validation_rows_put_ours_next_to_the_paper():
    summaries = paper_like_summaries()
    summaries["qwen"]["f1"] = 30.09
    row = next(r for r in validation_rows(summaries) if r["model"] == "qwen")
    assert row["f1_ours"] == pytest.approx(30.09)
    assert row["f1_paper"] == pytest.approx(29.09)
    assert row["f1_delta"] == pytest.approx(1.0)
    assert [r["model"] for r in validation_rows(summaries)] == [*OPEN_MODELS, "hallucinated"]


def test_country_checks_can_use_the_second_measure():
    summaries = paper_like_summaries()
    summaries["qwen"]["country_acc"] = 60.0  # strict measure misses
    assert failed(check_g1(summaries, country_key="country_acc_match")) == []
    assert failed(check_country(summaries, "country_acc")) == [
        "qwen country accuracy within 3 points"
    ]


def test_country_check_on_a_measure_that_was_not_computed_is_an_error():
    summaries = paper_like_summaries()
    del summaries["qwen"]["country_acc_match"]
    with pytest.raises(KeyError):
        check_country(summaries, "country_acc_match")
