import json
from pathlib import Path

import pytest

from georc.data import (
    RELEASED_TAGS,
    load_pairs,
    load_rounds,
    parse_expert_rounds,
    released_chain_path,
)
from georc.points import extract_points

DATA = Path(__file__).resolve().parents[1] / "data" / "georc"


def test_extract_points_strips_list_markers():
    chain = "1. Cyrillic on the sign\n* Painted tree trunks\n- North-south street grid"
    assert extract_points(chain) == [
        "Cyrillic on the sign",
        "Painted tree trunks",
        "North-south street grid",
    ]


def test_extract_points_drops_blank_lines_and_fragments_of_five_chars_or_fewer():
    chain = "Round\n\n   \nOk.\n12345\nLeft-hand traffic"
    assert extract_points(chain) == ["Left-hand traffic"]


def test_extract_points_keeps_markdown_inside_the_line():
    # The paper's scorer only strips a leading marker, so bold headings survive as points.
    assert extract_points("1. **Language**: Cyrillic script") == ["**Language**: Cyrillic script"]


def test_parse_expert_rounds_splits_on_round_headers():
    text = "Round 1\nCyrillic signage\nConclusion\nKyrgyzstan\nRound 2\nLeft-hand traffic\n"
    rounds = parse_expert_rounds(text)
    assert set(rounds) == {1, 2}
    assert extract_points(rounds[2]) == ["Left-hand traffic"]


def test_parse_expert_rounds_drops_lines_mentioning_reasoning():
    text = "Round 1\nReasoning\nCyrillic signage\nMy reasoning here is weak\nWhite tree trunks\n"
    assert extract_points(parse_expert_rounds(text)[1]) == ["Cyrillic signage", "White tree trunks"]


def test_parse_expert_rounds_stops_at_conclusion():
    text = "Round 1\nCyrillic signage\nConclusion\nThis is Kyrgyzstan\nTrailing line\n"
    assert extract_points(parse_expert_rounds(text)[1]) == ["Cyrillic signage"]


def _write_challenge(root: Path, name: str, countries: list[str], experts: dict[str, str]) -> None:
    folder = root / name
    folder.mkdir(parents=True)
    meta = [{"streakLocationCode": c, "lat": 0.0, "lng": 0.0} for c in countries]
    (folder / f"{name}_rounds_metadata.json").write_text(json.dumps(meta))
    for expert, text in experts.items():
        (folder / f"{expert}.txt").write_text(text)


def test_load_rounds_reads_lower_case_country_codes_in_round_order(tmp_path):
    _write_challenge(tmp_path, "abc", ["KG", "my"], experts={})
    rounds = load_rounds(tmp_path)
    assert [(r.challenge, r.index, r.country) for r in rounds] == [
        ("abc", 1, "kg"),
        ("abc", 2, "my"),
    ]
    assert rounds[0].image == tmp_path / "abc" / "abc_1.png"


def test_load_rounds_fails_loudly_on_a_missing_country_without_override(tmp_path):
    _write_challenge(tmp_path, "abc", [None], experts={})
    with pytest.raises(ValueError, match="abc round 1"):
        load_rounds(tmp_path)


def test_load_pairs_gives_one_pair_per_expert_per_round(tmp_path):
    two_rounds = "Round 1\nCyrillic signage\nRound 2\nLeft-hand traffic\n"
    _write_challenge(
        tmp_path,
        "abc",
        ["kg", "my"],
        experts={"Human_Expert_1": two_rounds, "Human_Expert_3": two_rounds},
    )
    pairs = load_pairs(tmp_path, load_rounds(tmp_path))
    assert [(p.round.index, p.expert) for p in pairs] == [
        (1, "Human_Expert_1"),
        (1, "Human_Expert_3"),
        (2, "Human_Expert_1"),
        (2, "Human_Expert_3"),
    ]


def test_released_chain_path_for_models_and_baselines(tmp_path):
    _write_challenge(tmp_path, "abc", ["kg"], experts={})
    rnd = load_rounds(tmp_path)[0]
    folder = tmp_path / "abc"
    qwen = released_chain_path(tmp_path, rnd, "qwen")
    hallucinated = released_chain_path(tmp_path, rnd, "hallucinated")
    assert qwen == folder / "candidate_reasoning_chain_qwen_1.txt"
    assert hallucinated == folder / "hallucinated_reasoning_chain_1.txt"
    assert len(RELEASED_TAGS) == 12


real_data = pytest.mark.skipif(not DATA.exists(), reason="GeoRC text data not in data/georc")


@real_data
def test_real_dataset_has_500_rounds_and_800_expert_pairs():
    rounds = load_rounds(DATA)
    assert len(rounds) == 500
    assert len(load_pairs(DATA, rounds)) == 800


@real_data
def test_real_dataset_ground_truth_countries_for_a_known_challenge():
    rounds = [r for r in load_rounds(DATA) if r.challenge == "1NJsXTxIF9GGMDxC"]
    assert [r.country for r in rounds] == ["kg", "my", "rs", "ar", "za"]


@real_data
def test_real_dataset_round_with_null_code_gets_the_documented_override():
    rnd = next(r for r in load_rounds(DATA) if r.key == "8Uo6ejwXYqmp9av3_4")
    assert rnd.country == "kr"


@real_data
@pytest.mark.parametrize("tag", ["gemma3_12b", "qwen"])
def test_real_dataset_has_a_released_chain_for_every_round(tag):
    rounds = load_rounds(DATA)
    assert all(released_chain_path(DATA, r, tag).exists() for r in rounds)
