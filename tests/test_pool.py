import json

import pytest

from georc.judge import FunctionJudge
from georc.pool import EvidenceSet, ask, ask_truth, clusters, extract_sets, run, score_chain

EXPERTS = {
    "Human_Expert_1": "Round 1\nCyrillic signage\nWhite tree trunks\nConclusion\nKyrgyzstan\n"
    "Round 2\nLeft-hand traffic\nConclusion\nMalaysia\n",
    "Human_Expert_2": "Round 1\nCyrillic signage\nConclusion\nRussia\n"
    "Round 2\nChevron design\nConclusion\nMalaysia\n",
    "Human_Expert_3": "Round 1\nGrid street plan\nConclusion\nKyrgyzstan\n"
    "Round 2\nLeft-hand traffic\nJungle roadside\nConclusion\nIndonesia\n",
}


@pytest.fixture
def dataset(tmp_path):
    """Challenge abc: two rounds, three experts. Challenge def: one round, one expert."""
    data, out = tmp_path / "data", tmp_path / "out"
    solo = {"Human_Expert_1": "Round 1\nYellow outer lines\nConclusion\nSA\n"}
    for name, n_rounds, experts in [("abc", 2, EXPERTS), ("def", 1, solo)]:
        folder = data / name
        folder.mkdir(parents=True)
        meta = [{"streakLocationCode": "kg", "lat": 1.0, "lng": 2.0}] * n_rounds
        (folder / f"{name}_rounds_metadata.json").write_text(json.dumps(meta))
        for expert, text in experts.items():
            (folder / f"{expert}.txt").write_text(text)
        for i in range(1, n_rounds + 1):
            chain = "1. Cyrillic signage\n2. Left-hand traffic\nConclusion: Kyrgyzstan"
            (folder / f"candidate_reasoning_chain_qwen_{i}.txt").write_text(chain)
    return data, out


def es(author: str, cue: str, location: str | None = None, weight: float = 1.0) -> EvidenceSet:
    return EvidenceSet("r", author, cue, location, None, weight)


def same(*pairs: tuple[str, str]) -> dict[tuple[str, str], bool]:
    return {key: True for a, b in pairs for key in ((a, b), (b, a))}


def test_clusters_group_expert_cues_that_were_judged_the_same():
    sets = [es("E1", "hills"), es("E2", "hilly terrain"), es("E2", "mint box"), es("E3", "slopes")]
    cue = same(("hills", "hilly terrain"), ("hilly terrain", "slopes"))
    assert clusters(sets, cue) == [[0, 1, 3], [2]]


def test_score_chain_gives_cue_place_and_coverage_marks():
    experts = [
        es("E1", "hills", "New Zealand", 0.5),
        es("E2", "hilly terrain", "California", 0.5),
        es("E2", "mint box", "US", 0.5),
    ]
    candidate = [es("m", "rolling hills", "California", 0.75), es("m", "a church", "Malta", 0.25)]
    cue = same(
        ("hills", "hilly terrain"), ("hills", "rolling hills"), ("hilly terrain", "rolling hills")
    )
    place = {("New Zealand", "California"): False, ("California", "California"): True}
    marks = score_chain(candidate, experts, cue, place, {})
    assert marks["cue"] == pytest.approx(75.0)  # the church is not in the pool
    assert marks["place"] == pytest.approx(50.0)  # half the pool's weight on hills says California
    assert marks["coverage"] == pytest.approx(50.0)  # hills cluster hit, mint box missed
    assert marks["reason"] is None


def test_score_chain_with_no_sets_scores_zero():
    marks = score_chain([], [es("E1", "hills")], {}, {}, {})
    assert (marks["cue"], marks["coverage"], marks["place"]) == (0.0, 0.0, None)


def test_ask_parses_yes_and_no_and_stores_both_orders():
    judge = FunctionJudge(lambda p: "Yes." if "FIRST: a\nSECOND: b" in p else "no")
    verdicts = ask("same_cue.txt", [("a", "b"), ("a", "c")], judge)
    assert verdicts[("a", "b")] is True and verdicts[("b", "a")] is True
    assert verdicts[("a", "c")] is False


def fake_judge(prompt: str) -> str:
    if "STATEMENT:\n" in prompt:  # extraction: one set per statement, the statement as the cue
        statement = prompt.rsplit("STATEMENT:\n", 1)[1]
        return json.dumps([{"cue": statement, "location": "L", "inference": "i", "weight": 100}])
    first, second = prompt.rsplit("FIRST: ", 1)[1].split("\nSECOND: ")
    return "yes" if first == second else "no"


def test_extract_sets_spreads_a_chains_weight_over_its_statements(dataset):
    result = run(*dataset, FunctionJudge(fake_judge), ["qwen"])
    sets = extract_sets(result["chains"]["qwen"], FunctionJudge(fake_judge))
    first = [s for s in sets if s.round == "abc_1"]
    assert [s.cue for s in first] == [
        "Cyrillic signage",
        "Left-hand traffic",
        "Conclusion: Kyrgyzstan",
    ]
    assert [s.weight for s in first] == [pytest.approx(1 / 3)] * 3


def test_run_scores_experts_leave_one_out_and_models_on_shared_rounds(dataset):
    result = run(*dataset, FunctionJudge(fake_judge), ["qwen"])
    assert len(result["chains"]["qwen"]) == 2  # only the two rounds with three experts
    rows = {(r["set"], r["round"], r["author"]): r for r in result["rows"]}
    e1 = rows[("experts", "abc_1", "Human_Expert_1")]  # Cyrillic (shared with E2), white trunks
    assert (e1["cue"], e1["coverage"]) == (pytest.approx(50.0), pytest.approx(50.0))
    qwen = rows[("qwen", "abc_1", "qwen")]  # of its three statements only Cyrillic is in the pool
    assert qwen["cue"] == pytest.approx(100 / 3)
    assert qwen["coverage"] == pytest.approx(100 / 3)  # pool: Cyrillic, white trunks, grid plan
    assert {r["set"] for r in result["rows"]} == {"experts", "qwen", "random"}
    assert result["summary"]["experts"]["n"] == 6


def test_ask_both_orders_needs_a_yes_in_each_direction():
    replies = {
        "FIRST: a\nSECOND: b": "yes",
        "FIRST: b\nSECOND: a": "no",
        "FIRST: a\nSECOND: c": "yes",
    }
    judge = FunctionJudge(lambda p: next((v for k, v in replies.items() if k in p), "yes"))
    verdicts = ask("same_cue.txt", [("a", "b"), ("a", "c")], judge, both_orders=True)
    assert verdicts[("a", "b")] is False and verdicts[("b", "a")] is False
    assert verdicts[("a", "c")] is True


def test_run_takes_the_cue_question_as_a_parameter(dataset):
    asked = []

    def judge(prompt: str) -> str:
        asked.append(prompt)
        return fake_judge(prompt)

    run(
        *dataset,
        FunctionJudge(judge),
        ["qwen"],
        cue_template="same_cue_strict.txt",
        both_orders=True,
    )
    assert any("SAME DISTINGUISHING DETAIL" in p for p in asked)


def test_run_can_match_with_a_different_judge_than_it_splits_with(dataset):
    split_prompts, match_prompts = [], []
    splitter = FunctionJudge(lambda p: split_prompts.append(p) or fake_judge(p))
    matcher = FunctionJudge(lambda p: match_prompts.append(p) or fake_judge(p))
    run(*dataset, splitter, ["qwen"], match_judge=matcher)
    assert split_prompts and all("STATEMENT:\n" in p for p in split_prompts)
    assert match_prompts and all("FIRST: " in p for p in match_prompts)


def test_ask_truth_checks_each_distinct_guess_once_against_the_true_location():
    asked = []

    def judge(prompt: str) -> str:
        asked.append(prompt)
        return "yes" if "GUESS: California" in prompt else "No"

    sets = [es("E1", "hills", "California"), es("E2", "homes", "California"), es("E1", "box")]
    sets.append(es("E3", "trees", "New Zealand"))
    verdicts = ask_truth(sets, {"r": (38.32, -123.03, "us")}, FunctionJudge(judge))
    assert verdicts == {("r", "California"): True, ("r", "New Zealand"): False}
    assert len(asked) == 2
    assert "latitude 38.32, longitude -123.03" in asked[0] and "country code US" in asked[0]
