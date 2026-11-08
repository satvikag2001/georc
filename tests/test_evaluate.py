import csv
import json

import pytest

from georc.data import load_pairs, load_rounds
from georc.evaluate import directory_chain_set, evaluate, released_chain_set, write_rows
from georc.judge import FunctionJudge


@pytest.fixture
def tiny_dataset(tmp_path):
    folder = tmp_path / "abc"
    folder.mkdir()
    meta = [{"streakLocationCode": "kg"}, {"streakLocationCode": "my"}]
    (folder / "abc_rounds_metadata.json").write_text(json.dumps(meta))
    (folder / "Human_Expert_1.txt").write_text(
        "Round 1\nCyrillic signage\nRound 2\nLeft-hand traffic\n"
    )
    (folder / "candidate_reasoning_chain_qwen_1.txt").write_text("Cyrillic signage")
    (folder / "candidate_reasoning_chain_qwen_2.txt").write_text("Snow poles")
    return tmp_path


def judge_that_knows_kyrgyzstan() -> FunctionJudge:
    def reply(prompt: str) -> str:
        if prompt.startswith("Below is a reasoning chain"):
            return '{"country": "KG"}'
        second = prompt.rsplit("SECOND REASONING CHAIN:", 1)[1].strip()
        first = prompt.rsplit("SECOND REASONING CHAIN:", 1)[0]
        return '{"precision": 100}' if second in first else '{"precision": 0}'

    return FunctionJudge(reply)


def test_evaluate_released_chains_gives_pair_and_round_rows(tiny_dataset):
    rounds = load_rounds(tiny_dataset)
    chains = released_chain_set(tiny_dataset, rounds, "qwen")
    pair_rows, round_rows, summary = evaluate(
        chains, load_pairs(tiny_dataset, rounds), rounds, judge_that_knows_kyrgyzstan()
    )
    assert [(r["round"], r["f1"]) for r in pair_rows] == [("abc_1", 100.0), ("abc_2", 0.0)]
    assert [(r["round"], r["guess"], r["correct"]) for r in round_rows] == [
        ("abc_1", "kg", 1),
        ("abc_2", "kg", 0),
    ]
    assert summary["f1"] == 50.0
    assert summary["country_acc"] == 50.0


def test_directory_chain_set_refuses_an_incomplete_folder(tiny_dataset, tmp_path):
    rounds = load_rounds(tiny_dataset)
    chains = tmp_path / "chains"
    chains.mkdir()
    (chains / "abc_1.txt").write_text("Cyrillic signage")
    with pytest.raises(FileNotFoundError, match="1 chains missing"):
        directory_chain_set(chains, rounds)


def test_write_rows_round_trips_through_csv(tmp_path):
    path = tmp_path / "out" / "rows.csv"
    write_rows(path, [{"round": "abc_1", "f1": 40.0}])
    with path.open() as f:
        assert list(csv.DictReader(f)) == [{"round": "abc_1", "f1": "40.0"}]
