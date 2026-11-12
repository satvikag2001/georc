import random

import pytest

from georc.stats import compare, paired_bootstrap, win


def planted(effect: float, images: int = 300) -> dict[str, list[float]]:
    rng = random.Random(1)
    return {f"img{i}": [effect + rng.gauss(0, 5) for _ in range(1 + i % 3)] for i in range(images)}


def test_bootstrap_recovers_a_planted_effect():
    mean, low, high = paired_bootstrap(planted(2.0), resamples=2000)
    assert low < 2.0 < high
    assert low > 0
    assert mean == pytest.approx(2.0, abs=0.6)


def test_bootstrap_interval_covers_zero_when_there_is_no_effect():
    _, low, high = paired_bootstrap(planted(0.0), resamples=2000)
    assert low < 0 < high


def test_bootstrap_resamples_images_not_pairs():
    # Three identical pairs on one image carry one image's worth of evidence, not three.
    one = {f"i{k}": [float(k % 2)] for k in range(40)}
    tripled = {image: values * 3 for image, values in one.items()}
    assert paired_bootstrap(tripled, resamples=500) == paired_bootstrap(one, resamples=500)


def test_bootstrap_is_reproducible():
    assert paired_bootstrap(planted(1.0), resamples=300) == paired_bootstrap(
        planted(1.0), resamples=300
    )


@pytest.mark.parametrize(
    ("f1_low", "country_low", "expected"),
    [
        (0.01, -1.99, True),
        (0.0, 5.0, False),  # the F1 interval must exclude 0
        (-0.01, 5.0, False),
        (3.0, -2.0, False),  # the country lower bound must be above -2 points
        (3.0, -2.01, False),
    ],
)
def test_win_rule_on_both_sides_of_each_threshold(f1_low, country_low, expected):
    assert win(f1_low, country_low) is expected


def test_compare_pairs_rows_by_round_and_expert_and_clusters_by_round():
    base = [
        {"round": "a_1", "expert": "E1", "f1": "10"},
        {"round": "a_1", "expert": "E2", "f1": "20"},
        {"round": "b_1", "expert": "E1", "f1": "30"},
    ]
    treat = [  # different order on purpose
        {"round": "b_1", "expert": "E1", "f1": "36"},
        {"round": "a_1", "expert": "E2", "f1": "23"},
        {"round": "a_1", "expert": "E1", "f1": "13"},
    ]
    assert compare(base, treat, "f1", ("round", "expert")) == {"a_1": [3.0, 3.0], "b_1": [6.0]}


def test_compare_refuses_rows_that_do_not_line_up():
    base = [{"round": "a_1", "expert": "E1", "f1": "1"}]
    with pytest.raises(ValueError, match="same"):
        compare(base, [{"round": "a_2", "expert": "E1", "f1": "1"}], "f1", ("round", "expert"))
