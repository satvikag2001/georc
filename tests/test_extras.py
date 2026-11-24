import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from georc.data import load_pairs, load_rounds
from georc.extras import (
    EXPERTS,
    PASSES,
    SETS,
    CacheOnly,
    Chain,
    Place,
    build_evidence_prompt,
    build_location_prompt,
    build_specificity_prompt,
    cache_path,
    call_counts,
    describe,
    evidence_pass,
    experts_pass,
    grounded_pass,
    grounded_request,
    load_set,
    location_pass,
    pairwise_pass,
    paper_expert_pair,
    parse_evidence,
    parse_location,
    parse_specificity,
    parse_verdicts,
    run_pass,
    specificity_pass,
    split_request,
)
from georc.judge import CachedJudge, FunctionJudge

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "georc"
real_data = pytest.mark.skipif(not DATA.exists(), reason="GeoRC text data not in data/georc")

EXPERT_FILES = {
    "Human_Expert_1": "Round 1\nCyrillic signage\nWhite tree trunks\nConclusion\nKyrgyzstan\n"
    "Round 2\nLeft-hand traffic\nConclusion\nMalaysia\n",
    "Human_Expert_2": "Round 1\nCyrillic signage\nConclusion\nRussia\n"
    "Round 2\nChevron design\nConclusion\nMalaysia\n",
    "Human_Expert_3": "Round 1\nGrid street plan\nConclusion\nKyrgyzstan\n"
    "Round 2\nLeft-hand traffic\nJungle roadside\nConclusion\nIndonesia\n",
}


@pytest.fixture
def dataset(tmp_path) -> tuple[Path, Path]:
    """Challenge abc: two rounds, three experts each. Challenge def: one round, one expert."""
    data, out = tmp_path / "data", tmp_path / "out"
    for name, countries, experts in [
        ("abc", ["kg", "my"], EXPERT_FILES),
        ("def", ["za"], {"Human_Expert_1": "Round 1\nYellow outer lines\nConclusion\nSA\n"}),
    ]:
        folder = data / name
        folder.mkdir(parents=True)
        meta = [{"streakLocationCode": c, "lat": 1.0, "lng": 2.0} for c in countries]
        (folder / f"{name}_rounds_metadata.json").write_text(json.dumps(meta))
        for expert, text in experts.items():
            (folder / f"{expert}.txt").write_text(text)
        for i in range(1, len(countries) + 1):
            chain = "1. Cyrillic signage\n2. Left-hand traffic\nConclusion: Kyrgyzstan"
            (folder / f"candidate_reasoning_chain_qwen_{i}.txt").write_text(chain)
            generated = out / "chains" / "G0" / "seed0" / f"{name}_{i}.txt"
            generated.parent.mkdir(parents=True, exist_ok=True)
            generated.write_text("Snow poles\nConclusion: Norway")
    return data, out


def split_prompt(prompt: str) -> tuple[str, str]:
    head, second = prompt.rsplit("SECOND REASONING CHAIN:", 1)
    return head.rsplit("FIRST REASONING CHAIN:", 1)[1].strip(), second.strip()


def any_reply(prompt: str) -> str:
    """A deterministic reply every parser can read, varying with the prompt."""
    n = int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 101
    obj = {"precision": n, "specificity": n, "country": "KG", "place": "P", "lat": n, "lon": -n}
    return json.dumps(obj) + "\n1: yes\n2: no\n3: na"


# Chain sets and pairs


def test_expert_chains_are_cleaned_and_keep_the_full_round_text_for_the_guess(dataset):
    chains, _ = load_set(EXPERTS, *dataset)
    assert [(c.round.key, c.author) for c in chains][:3] == [
        ("abc_1", "Human_Expert_1"),
        ("abc_1", "Human_Expert_2"),
        ("abc_1", "Human_Expert_3"),
    ]
    assert len(chains) == 7
    first = chains[0]
    assert "Conclusion" not in first.text and "Kyrgyzstan" not in first.text
    assert first.full == "Cyrillic signage\nWhite tree trunks\nConclusion\nKyrgyzstan"


def test_expert_pairs_are_every_ordered_pair_of_different_experts_in_a_round(dataset):
    _, pairs = load_set(EXPERTS, *dataset)
    keys = [(c.round.key, c.author, ref.expert) for c, ref in pairs]
    assert len(keys) == len(set(keys)) == 12  # 2 rounds x 3 x 2; the one-expert round gives none
    assert all(author != expert for _, author, expert in keys)
    assert all(c.round == ref.round for c, ref in pairs)
    reference = next(ref for c, ref in pairs if ref.expert == "Human_Expert_2")
    assert "Conclusion" not in reference.chain  # references are the cleaned chains


def test_model_set_pairs_follow_the_existing_one_to_all_order(dataset):
    data, out = dataset
    chains, pairs = load_set("qwen", data, out)
    expected = [(p.round.key, p.expert) for p in load_pairs(data, load_rounds(data))]
    assert [(c.round.key, ref.expert) for c, ref in pairs] == expected
    assert {c.author for c in chains} == {"qwen"}
    assert chains[0].text == chains[0].full


def test_generated_sets_are_read_from_their_seed0_chain_folder(dataset):
    chains, _ = load_set("G0", *dataset)
    assert [c.text for c in chains] == ["Snow poles\nConclusion: Norway"] * 3


@pytest.mark.parametrize(
    ("author", "expert", "paper"),
    [(1, 2, True), (2, 3, True), (1, 3, True), (2, 1, False), (3, 2, False), (3, 1, False)],
)
def test_paper_pairs_take_the_first_named_expert_as_the_candidate(author, expert, paper):
    assert paper_expert_pair(f"Human_Expert_{author}", f"Human_Expert_{expert}") is paper


@real_data
def test_real_experts_set_has_900_ordered_pairs_on_150_rounds_and_450_paper_pairs():
    chains, pairs = load_set(EXPERTS, DATA, Path("unused"))
    assert len(chains) == 800 and len(pairs) == 900
    assert len({c.round.key for c, _ in pairs}) == 150
    assert set(Counter(c.round.key for c, _ in pairs).values()) == {6}
    combos = Counter((c.author, ref.expert) for c, ref in pairs)
    assert len(combos) == 6 and set(combos.values()) == {150}
    assert sum(paper_expert_pair(c.author, ref.expert) for c, ref in pairs) == 450


def test_every_pass_and_set_has_its_own_cache_file(tmp_path):
    paths = {cache_path(tmp_path, p, s) for p in PASSES for s in SETS}
    assert len(paths) == len(PASSES) * len(SETS)
    assert cache_path(tmp_path, "pairwise", "qwen") == (
        tmp_path / "judge_cache" / "extras" / "pairwise__qwen.jsonl"
    )


# experts


def test_experts_pass_scores_ordered_pairs_and_reads_the_country_from_the_full_text(dataset):
    chains, pairs = load_set(EXPERTS, *dataset)

    def reply(prompt: str) -> str:
        if prompt.startswith("Below is a reasoning chain"):
            return '{"country": "KG"}' if "Kyrgyzstan" in prompt else '{"country": "XX"}'
        first, second = split_prompt(prompt)
        return '{"precision": 100}' if second in first.split(" || ") else '{"precision": 0}'

    pair_rows, chain_rows = experts_pass(chains, pairs, FunctionJudge(reply))
    row = next(
        r
        for r in pair_rows
        if (r["round"], r["author"], r["expert"]) == ("abc_1", "Human_Expert_1", "Human_Expert_2")
    )
    assert (row["precision"], row["recall"], row["paper"]) == (50.0, 100.0, True)
    back = next(
        r
        for r in pair_rows
        if (r["round"], r["author"], r["expert"]) == ("abc_1", "Human_Expert_2", "Human_Expert_1")
    )
    assert (back["precision"], back["recall"], back["paper"]) == (100.0, 50.0, False)
    assert [(r["round"], r["author"], r["guess"], r["correct"], r["shared"]) for r in chain_rows][
        :3
    ] == [
        ("abc_1", "Human_Expert_1", "kg", 1, 1),
        ("abc_1", "Human_Expert_2", "", 0, 1),  # "Russia" is no Kyrgyzstan
        ("abc_1", "Human_Expert_3", "kg", 1, 1),
    ]
    assert chain_rows[-1]["shared"] == 0  # the one-expert round


def test_run_pass_refuses_the_experts_pass_on_a_model_set(dataset):
    with pytest.raises(ValueError, match="experts"):
        run_pass("experts", "qwen", *dataset, FunctionJudge(any_reply))


# pairwise


def test_pairwise_gives_an_expert_points_by_candidate_points_matrix(dataset):
    _, pairs = load_set("qwen", *dataset)
    prompts = []

    def reply(prompt: str) -> str:
        prompts.append(prompt)
        first, second = split_prompt(prompt)
        return '{"precision": 90}' if first == second else '{"precision": 10}'

    [row] = pairwise_pass(pairs[:1], FunctionJudge(reply))
    assert row["expert_points"] == ["Cyrillic signage", "White tree trunks"]
    assert row["candidate_points"] == [
        "Cyrillic signage",
        "Left-hand traffic",
        "Conclusion: Kyrgyzstan",
    ]
    assert row["scores"] == [[90.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
    assert len(prompts) == 6
    assert all("Only award points if there is logical evidence" in p for p in prompts)  # repo


def test_pairwise_marks_unparsed_cells_as_none(dataset):
    _, pairs = load_set("qwen", *dataset)
    [row] = pairwise_pass(pairs[:1], FunctionJudge(lambda p: "no idea"))
    assert row["scores"] == [[None] * 3] * 2


# specificity


def test_specificity_prompt_ends_with_the_statement_and_asks_for_json():
    prompt = build_specificity_prompt("Sign reads {EXIT}")
    assert prompt.endswith("STATEMENT: Sign reads {EXIT}")
    assert '{"specificity": <score>}' in prompt


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"specificity": 75}', 75.0),
        ('Sure. {"specificity": 12.5}', 12.5),
        ('{"specificity": 0}', 0.0),
        ('{"specificity": 100}', 100.0),
        ('{"specificity": 140}', None),
        ('{"specificity": -5}', None),
        ('{"specificity": "high"}', None),
        ('{"precision": 75}', None),
        ("75", None),
    ],
)
def test_parse_specificity_rejects_out_of_range_and_missing_scores(reply, expected):
    assert parse_specificity(reply) == expected


def test_specificity_judges_each_unique_point_once(dataset):
    chains, _ = load_set("qwen", *dataset)  # three identical chains
    prompts = []

    def reply(prompt: str) -> str:
        prompts.append(prompt)
        return '{"specificity": 50}'

    result = specificity_pass(chains, FunctionJudge(reply))
    assert result == {
        "Cyrillic signage": 50.0,
        "Left-hand traffic": 50.0,
        "Conclusion: Kyrgyzstan": 50.0,
    }
    assert len(prompts) == 3


# location


def test_location_prompt_ends_with_the_chain():
    prompt = build_location_prompt("Conclusion: Bishkek")
    assert prompt.endswith("REASONING CHAIN: Conclusion: Bishkek")
    assert '"lat"' in prompt and '"lon"' in prompt


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"place": "Bishkek", "lat": 42.87, "lon": 74.59}', Place("Bishkek", 42.87, 74.59)),
        (
            '```json\n{"place": "Malaysia", "lat": 4, "lon": 102}\n```',
            Place("Malaysia", 4.0, 102.0),
        ),
        ('{"place": null, "lat": null, "lon": null}', Place(None, None, None)),
        ('{"place": "Kyrgyzstan", "lat": null, "lon": null}', Place("Kyrgyzstan", None, None)),
        ("Bishkek, 42.87, 74.59", None),
        ('{"place": Bishkek, "lat": 42.87, "lon": 74.59}', None),
        ('{"place": "X", "lat": 95, "lon": 0}', None),
        ('{"place": "X", "lat": 0, "lon": -181}', None),
        ('{"place": "X", "lat": 42.8}', None),
        ('{"place": "X", "lat": "42", "lon": "74"}', None),
        ('{"place": "X", "lat": true, "lon": 1}', None),
        ('{"place": 7, "lat": 1, "lon": 1}', None),
    ],
)
def test_parse_location(reply, expected):
    assert parse_location(reply) == expected


def test_location_reads_the_experts_full_text_with_the_conclusion(dataset):
    chains, _ = load_set(EXPERTS, *dataset)
    prompts = []

    def reply(prompt: str) -> str:
        prompts.append(prompt)
        return '{"place": "Bishkek", "lat": 42.87, "lon": 74.59}'

    rows = location_pass(chains[:1], FunctionJudge(reply))
    assert prompts[0].endswith("Conclusion\nKyrgyzstan")
    assert rows == [
        {
            "round": "abc_1",
            "author": "Human_Expert_1",
            "parsed": True,
            "place": "Bishkek",
            "lat": 42.87,
            "lon": 74.59,
        }
    ]


# grounded


def test_grounded_request_names_the_round_then_numbers_the_statements():
    key, prompt = split_request(grounded_request("abc_1", ["Cyrillic signage", "Painted trunks"]))
    assert key == "abc_1"
    assert prompt.endswith("STATEMENTS:\n1. Cyrillic signage\n2. Painted trunks")
    assert '"1: yes"' in prompt


def test_grounded_cache_key_depends_on_the_image_as_well_as_the_text(tmp_path):
    seen = []
    judge = CachedJudge(FunctionJudge(lambda p: seen.append(p) or "1: yes"), tmp_path / "c.jsonl")
    judge.complete(
        [grounded_request("abc_1", ["Cyrillic"]), grounded_request("abc_2", ["Cyrillic"])]
    )
    assert len(seen) == 2


@pytest.mark.parametrize(
    ("reply", "n", "expected"),
    [
        ("1: yes\n2: no\n3: na", 3, ["yes", "no", "na"]),
        ("1: Yes\n2. NO\n3) n/a", 3, ["yes", "no", "na"]),
        ("**1:** yes, the sign is visible\n- 2: no", 2, ["yes", "no"]),
        ("2: no\n1: yes", 2, ["yes", "no"]),
        ("1: yes\n3: no", 3, ["yes", None, "no"]),
        ("1: yes\n1: no", 1, ["yes"]),
        ("1: yes\n7: no", 2, ["yes", None]),
        ("1: not visible", 1, [None]),
        ("", 2, [None, None]),
    ],
)
def test_parse_verdicts_counts_missing_verdicts_as_none(reply, n, expected):
    assert parse_verdicts(reply, n) == expected


def test_grounded_pass_gives_one_verdict_per_point_and_skips_empty_chains(dataset):
    chains, _ = load_set(EXPERTS, *dataset)
    empty = Chain(chains[0].round, "Human_Expert_9", "", "")
    requests = []

    def reply(request: str) -> str:
        requests.append(request)
        return "1: yes\n2: no"

    rows = grounded_pass([chains[0], empty], FunctionJudge(reply))
    assert len(requests) == 1 and split_request(requests[0])[0] == "abc_1"
    assert rows[0]["points"] == ["Cyrillic signage", "White tree trunks"]
    assert rows[0]["verdicts"] == ["yes", "no"]
    assert rows[1]["verdicts"] == []


# caching, dry runs and logs


@pytest.mark.parametrize("pass_name", PASSES)
def test_a_cache_only_rerun_reproduces_the_results(dataset, pass_name):
    data, out = dataset
    for set_name in [EXPERTS] if pass_name == "experts" else ["qwen", EXPERTS]:
        path = cache_path(out, pass_name, set_name)
        first = run_pass(
            pass_name, set_name, data, out, CachedJudge(FunctionJudge(any_reply), path)
        )
        again = run_pass(pass_name, set_name, data, out, CachedJudge(CacheOnly(), path))
        assert again == first


def test_cache_only_fails_on_a_prompt_that_was_never_judged(tmp_path):
    with pytest.raises(LookupError, match="1 prompts"):
        CachedJudge(CacheOnly(), tmp_path / "c.jsonl").complete(["p"])


def test_call_counts_dedupe_prompts_and_subtract_what_is_cached(tmp_path):
    path = tmp_path / "c.jsonl"
    CachedJudge(FunctionJudge(lambda p: "r"), path).complete(["a"])
    assert call_counts(["a", "b", "b", "c"], path) == {
        "calls": 4,
        "unique": 3,
        "cached": 1,
        "to_run": 2,
    }
    assert call_counts(["a"], tmp_path / "missing.jsonl")["to_run"] == 1


@pytest.mark.parametrize("pass_name", PASSES)
def test_describe_summarizes_every_pass(dataset, pass_name):
    set_name = EXPERTS if pass_name == "experts" else "qwen"
    result = run_pass(pass_name, set_name, *dataset, FunctionJudge(any_reply))
    text = describe(pass_name, set_name, result)
    assert text.startswith(f"{pass_name} {set_name}")
    if pass_name == "experts":
        assert "paper pairs" in text and "56.69" in text and "96.67" in text


def run_script(*args: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(REPO / "scripts" / "judge_extras.py"), *args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)


def test_dry_run_prints_call_counts_per_set_and_writes_no_cache(dataset):
    data, out = dataset
    common = ["--data", str(data), "--out", str(out), "--dry-run"]
    result = run_script(*common, "--pass", "pairwise", "--sets", "qwen", "G0")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        # 9 expert points (7 distinct) x 3 qwen points, and x 2 G0 points
        "pairwise qwen: calls=27 unique=21 cached=0 to_run=21",
        "pairwise G0: calls=18 unique=14 cached=0 to_run=14",
    ]
    assert not (out / "judge_cache").exists()
    experts = run_script(*common, "--pass", "experts")  # the experts set by default
    assert experts.stdout.startswith("experts experts: calls=")


def test_the_experts_pass_takes_only_the_experts_set(dataset):
    data, out = dataset
    result = run_script(
        "--data", str(data), "--out", str(out), "--pass", "experts", "--sets", "qwen"
    )
    assert result.returncode == 2 and "experts" in result.stderr


# evidence sets


def test_evidence_prompt_shows_the_chain_then_the_statement_and_survives_braces():
    prompt = build_evidence_prompt(
        ["Sign reads {EXIT}", "Red soil {statement}"], "Sign reads {EXIT}"
    )
    assert prompt.endswith("STATEMENT:\nSign reads {EXIT}")
    assert "CHAIN:\nSign reads {EXIT}\nRed soil {statement}\nSTATEMENT:" in prompt
    assert '"cue"' in prompt and '"weight"' in prompt


def test_parse_evidence_reads_sets_and_scales_weights_to_100():
    reply = (
        'Here: [{"cue": "hills", "location": "N. California", "inference": null, "weight": 30},'
        ' {"cue": "homes", "location": "N. California", "inference": "style", "weight": 90}]'
    )
    assert parse_evidence(reply) == [
        {"cue": "hills", "location": "N. California", "inference": None, "weight": 25.0},
        {"cue": "homes", "location": "N. California", "inference": "style", "weight": 75.0},
    ]


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("[]", []),
        ("no json here", None),
        ('{"cue": "hills"}', None),
        ("[1, 2]", None),
        ('[{"location": "India", "weight": 100}]', None),
        (
            '[{"cue": "blur", "location": "India"}, {"cue": "blur", "location": "Cambodia"}]',
            [
                {"cue": "blur", "location": "India", "inference": None, "weight": 50.0},
                {"cue": "blur", "location": "Cambodia", "inference": None, "weight": 50.0},
            ],
        ),
        (
            '[{"cue": "blur", "location": "India", "weight": 0}, {"cue": "sky", "weight": "x"}]',
            [
                {"cue": "blur", "location": "India", "inference": None, "weight": 50.0},
                {"cue": "sky", "location": None, "inference": None, "weight": 50.0},
            ],
        ),
    ],
)
def test_parse_evidence_handles_empty_bad_and_unweighted_replies(reply, expected):
    assert parse_evidence(reply) == expected


def test_evidence_pass_gives_one_row_per_statement_with_the_whole_chain_as_context(dataset):
    chains, _ = load_set(EXPERTS, *dataset)
    prompts = []

    def reply(prompt: str) -> str:
        prompts.append(prompt)
        return '[{"cue": "c", "location": "L", "inference": "i", "weight": 100}]'

    rows = evidence_pass(chains, FunctionJudge(reply))
    first = [r for r in rows if r["round"] == "abc_1" and r["author"] == "Human_Expert_1"]
    assert [(r["index"], r["statement"]) for r in first] == [
        (0, "Cyrillic signage"),
        (1, "White tree trunks"),
    ]
    assert first[0]["sets"] == [{"cue": "c", "location": "L", "inference": "i", "weight": 100.0}]
    asked = next(p for p in prompts if p.endswith("STATEMENT:\nWhite tree trunks"))
    assert "CHAIN:\nCyrillic signage\nWhite tree trunks\nSTATEMENT:" in asked
    assert "Kyrgyzstan" not in asked  # the conclusion is not part of the scored chain


def test_evidence_is_a_pass_and_describe_counts_sets():
    assert "evidence" in PASSES
    rows = [
        {"sets": [{"cue": "a"}, {"cue": "b"}]},
        {"sets": []},
        {"sets": None},
    ]
    assert describe("evidence", "experts", rows) == (
        "evidence experts: statements=3 with_sets=1 empty=1 unparsed=1 sets_per_statement=2.00"
    )
