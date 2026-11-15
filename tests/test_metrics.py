import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from georc.data import Pair, Round, chain_prefix, load_rounds
from georc.extras import EXPERTS, GENERATED, PASSES, RELEASED, Chain, cache_path, load_set, run_pass
from georc.judge import CachedJudge, FunctionJudge
from georc.metrics import (
    beat_rate,
    chain_records,
    collect,
    f1,
    grounded_precision,
    hard_f1,
    haversine_km,
    one_to_one,
    pair_records,
    rescale,
    summary_line,
    value,
    weighted_recall,
    win_units,
)
from georc.score import point_scores

REPO = Path(__file__).resolve().parents[1]

# Maths on hand-computed inputs


def test_f1_is_the_harmonic_mean_and_zero_without_signal():
    assert f1(50.0, 100.0) == pytest.approx(200 / 3)
    assert f1(0.0, 0.0) == 0.0


def test_hard_f1_counts_points_at_or_above_the_threshold():
    # P = 2/3 (50 counts, None does not), R = 1/2: F1 = 2 * (2/3 * 1/2) / (2/3 + 1/2) = 4/7
    assert hard_f1([60.0, 50.0, None], [70.0, 49.9], 50) == pytest.approx(400 / 7)
    assert hard_f1([60.0], [40.0], 50) == 0.0
    assert hard_f1([], [70.0], 50) == 0.0


def test_one_to_one_beats_greedy_matching():
    # Greedy takes 90 first and is left with 0 (total 90); the best assignment is 80 + 85.
    assert one_to_one([[90.0, 80.0], [85.0, 0.0]]) == pytest.approx((82.5, 82.5, 82.5))


def test_one_to_one_on_a_rectangle_divides_by_each_sides_point_count():
    p, r, _ = one_to_one([[10.0, 50.0, None]])  # one expert point, three candidate points
    assert (p, r) == pytest.approx((50 / 3, 50.0))
    assert one_to_one([[None, 30.0]])[:2] == pytest.approx((15.0, 30.0))  # unparsed counts 0
    assert one_to_one([]) == (0.0, 0.0, 0.0)


def test_weighted_recall_weights_expert_points_by_specificity():
    assert weighted_recall([100.0, 0.0], [75.0, 25.0]) == pytest.approx(75.0)
    assert weighted_recall([100.0, 0.0], [None, 50.0]) == 0.0  # unparsed weight counts 0
    assert weighted_recall([None, 80.0], [50.0, 50.0]) == pytest.approx(40.0)


def test_weighted_recall_falls_back_to_plain_recall_when_every_weight_is_zero():
    assert weighted_recall([100.0, 0.0], [0.0, 0.0]) == pytest.approx(50.0)


def test_grounded_precision_zeroes_only_points_judged_not_visible():
    verdicts = ["yes", "no", None, "na"]
    assert grounded_precision([80.0, 60.0, 40.0, None], verdicts) == pytest.approx(30.0)


def test_haversine_on_known_city_pairs():
    paris, london = (48.8566, 2.3522), (51.5074, -0.1278)
    assert haversine_km(paris, london) == pytest.approx(343.5, abs=1.0)
    new_york, los_angeles = (40.7128, -74.0060), (34.0522, -118.2437)
    assert haversine_km(new_york, los_angeles) == pytest.approx(3936, abs=2.0)
    assert haversine_km((0.0, 0.0), (0.0, 180.0)) == pytest.approx(math.pi * 6371.0)


def test_beat_rate_counts_ties_as_half():
    assert beat_rate([50.0, 70.0], 50.0) == 0.75
    assert beat_rate([40.0], 50.0) == 0.0


def test_win_units_average_over_the_other_experts_and_are_50_for_experts():
    expert_f1 = {
        ("r1", "E1", "E3"): 60.0,
        ("r1", "E2", "E3"): 40.0,
        ("r1", "E2", "E1"): 30.0,
        ("r1", "E3", "E1"): 30.0,
    }
    model = {("r1", "qwen", "E3"): 40.0, ("r1", "qwen", "E1"): 50.0, ("r9", "qwen", "E1"): 1.0}
    # vs E3: 60 beats 40, 40 ties -> 75; vs E1: both 30 lose to 50 -> 0; r9 has no experts
    assert win_units(model, expert_f1) == {"r1": [75.0, 0.0]}
    assert win_units(expert_f1, expert_f1) == {"r1": [50.0, 50.0]}


def test_rescale_puts_hallucinated_at_0_and_experts_at_100():
    assert rescale(40.0, 10.0, 70.0) == pytest.approx(50.0)
    assert rescale(40.0, 10.0, 10.0) is None
    assert rescale(None, 10.0, 70.0) is None


# Per-pair and per-chain records


def tiny_pair(author: str = "qwen") -> tuple[Chain, Pair]:
    rnd = Round("abc", 1, "kg", Path("abc_1.png"))
    chain = Chain(rnd, author, "Cyrillic signage\nSnow poles", "Cyrillic signage\nSnow poles")
    return chain, Pair(rnd, "Human_Expert_1", "Cyrillic signage\nWhite tree trunks")


def test_pair_records_gate_f1_on_the_candidates_country():
    pair = tiny_pair()
    points = [([100.0, 0.0], [100.0, 0.0])]  # P = R = F1 = 50
    [right] = pair_records([pair], points, correct={("abc_1", "qwen"): True})
    [wrong] = pair_records([pair], points, correct={("abc_1", "qwen"): False})
    assert (right["F1"], right["gated_F1"], wrong["gated_F1"]) == (50.0, 50.0, 0.0)


def test_pair_records_leave_out_metrics_whose_source_is_missing():
    [record] = pair_records([tiny_pair()], [([100.0, 0.0], [100.0, 0.0])])
    assert {"P", "R", "F1", "hard_F1_t50"} <= set(record)
    assert not {"o2o_F1", "spec_mean", "wspec_R", "grounded_P", "gated_F1"} & set(record)
    [bare] = pair_records([tiny_pair()], None)
    assert set(bare) == {"round", "author", "expert"}


def test_pair_records_join_every_source_on_the_same_points():
    pair = tiny_pair()
    [record] = pair_records(
        [pair],
        [([100.0, 0.0], [100.0, 0.0])],
        pairwise={("abc_1", "qwen", "Human_Expert_1"): [[90.0, 0.0], [0.0, 10.0]]},
        candidate_spec={"Cyrillic signage": 50.0, "Snow poles": None},
        expert_spec={"Cyrillic signage": 50.0, "White tree trunks": 0.0},
        verdicts={("abc_1", "qwen"): ["yes", "no"]},
    )
    assert record["o2o_F1"] == pytest.approx(50.0)  # (90 + 10) / 2 points each side
    assert record["spec_mean"] == 50.0  # the unparsed rating is skipped
    assert record["wspec_R"] == 100.0  # only the specific expert point counts, and it matched
    assert record["wspec_F1"] == pytest.approx(200 / 3)
    assert (record["grounded_rate"], record["no_visual_share"]) == (50.0, 0.0)
    assert record["grounded_P"] == 50.0  # the matched point was visible


def test_chain_records_give_errors_shares_and_the_country_split():
    truth = {"abc_1": (48.8566, 2.3522), "abc_2": (0.0, 0.0)}
    rows = [
        {"round": "abc_1", "author": "qwen", "parsed": True, "lat": 51.5074, "lon": -0.1278},
        {"round": "abc_2", "author": "qwen", "parsed": True, "lat": None, "lon": None},
        {"round": "abc_2", "author": "x", "parsed": False, "lat": None, "lon": None},
    ]
    correct = {("abc_1", "qwen"): False, ("abc_2", "qwen"): True}
    london, none = chain_records(rows, truth, correct)
    assert london["median_km"] == pytest.approx(343.5, abs=1.0)
    assert [london[f"within_{d}"] for d in (25, 200, 750, 2500)] == [0.0, 0.0, 100.0, 100.0]
    assert (london["no_guess"], london["median_km_wrong"]) == (0.0, london["median_km"])
    assert none["median_km"] == math.inf and none["no_guess"] == 100.0
    assert none["median_km_right"] == math.inf and "median_km_wrong" not in none


# Summary


def make_collected(values: dict[str, list[float]], metric: str = "m") -> dict:
    """Two rounds per set; values[set] = [round 1 value, round 2 value]."""
    return {name: {"shared": {metric: {"r1": [v[0]], "r2": [v[1]]}}} for name, v in values.items()}


def test_summary_line_places_sets_between_hallucinated_and_experts():
    collected = make_collected(
        {
            EXPERTS: [80.0, 60.0],
            "hallucinated": [10.0, 10.0],
            "qwen": [40.0, 40.0],
            "gemma3_12b": [30.0, 30.0],
            "llama3_11b": [20.0, 20.0],
            "qwen3": [25.0, 25.0],
            "G0": [50.0, 50.0],
            "A1": [55.0, 55.0],
        }
    )
    line = summary_line(collected, "m")
    assert (line["E"], line["H"]) == (70.0, 10.0)
    assert (line["weakest"], line["strongest"]) == ("llama3_11b", "qwen")
    assert line["pos_qwen"] == pytest.approx(50.0) and line["pos_G0"] == pytest.approx(200 / 3)
    assert line["hw_E_minus_strongest"] == pytest.approx(10.0)  # E - qwen is 40, 30 or 20
    assert line["hw_weakest_minus_H"] == 0.0
    assert line["ordered"] is True


def test_summary_line_flips_error_metrics_so_higher_is_better():
    km = {EXPERTS: 100.0, "hallucinated": 3000.0, "qwen": 500.0, "gemma3_12b": 900.0}
    km |= {"llama3_11b": 2000.0, "qwen3": 1500.0, "G0": 700.0, "A1": 600.0}
    line = summary_line(
        make_collected({k: [v, v] for k, v in km.items()}, "median_km"), "median_km"
    )
    assert (line["E"], line["H"]) == (-100.0, -3000.0)
    assert (line["strongest"], line["weakest"]) == ("qwen", "llama3_11b")
    assert line["ordered"] is True
    assert line["pos_qwen"] == pytest.approx(100 * 2500 / 2900)


def test_summary_line_reports_nothing_it_cannot_compute():
    partial = make_collected({EXPERTS: [80.0, 60.0], "qwen": [40.0, 40.0]})
    line = summary_line(partial, "m")
    assert line["E"] == 70.0 and line["H"] is None and line["ordered"] is None
    assert line["pos_qwen"] is None and line["hw_weakest_minus_H"] is None


def test_value_aggregates_by_mean_or_median_and_is_none_when_missing():
    collected = {"qwen": {"all": {"F1": {"r1": [10.0, 20.0], "r2": [60.0]}}}}
    collected["qwen"]["all"]["median_km"] = {"r1": [1.0, 2.0], "r2": [100.0]}
    assert value(collected, "qwen", "all", "F1") == 30.0
    assert value(collected, "qwen", "all", "median_km") == 2.0
    assert value(collected, "qwen", "all", "o2o_F1") is None
    assert value(collected, "A1", "all", "F1") is None


# Loading from caches, with fakes


def fake_reply(prompt: str) -> str:
    n = int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 101
    obj = {"precision": n, "specificity": n, "country": "KG", "place": "P", "lat": n - 50, "lon": n}
    return json.dumps(obj) + "\n1: yes\n2: no\n3: na"


@pytest.fixture
def runs_dataset(tmp_path) -> tuple[Path, Path]:
    """Challenge abc: two rounds, three experts each; def: one round, one expert; every set."""
    data, runs = tmp_path / "data", tmp_path / "runs"
    experts = {
        "Human_Expert_1": "Round 1\nCyrillic signage\nWhite tree trunks\nConclusion\nKyrgyzstan\n"
        "Round 2\nLeft-hand traffic\nConclusion\nMalaysia\n",
        "Human_Expert_2": "Round 1\nCyrillic signage\nConclusion\nRussia\n"
        "Round 2\nChevron design\nConclusion\nMalaysia\n",
        "Human_Expert_3": "Round 1\nGrid street plan\nConclusion\nKyrgyzstan\n"
        "Round 2\nLeft-hand traffic\nJungle roadside\nConclusion\nIndonesia\n",
    }
    for name, countries, files in [
        ("abc", ["kg", "my"], experts),
        ("def", ["za"], {"Human_Expert_1": "Round 1\nYellow outer lines\nConclusion\nSA\n"}),
    ]:
        folder = data / name
        folder.mkdir(parents=True)
        meta = [{"streakLocationCode": c, "lat": 10.0, "lng": 20.0} for c in countries]
        (folder / f"{name}_rounds_metadata.json").write_text(json.dumps(meta))
        for expert, text in files.items():
            (folder / f"{expert}.txt").write_text(text)
        for i in range(1, len(countries) + 1):
            for tag in RELEASED:
                chain = f"1. Cyrillic signage\n2. {tag} detail\nConclusion: Kyrgyzstan"
                (folder / f"{chain_prefix(tag)}{i}.txt").write_text(chain)
            for condition in GENERATED:
                generated = runs / "chains" / condition / "seed0" / f"{name}_{i}.txt"
                generated.parent.mkdir(parents=True, exist_ok=True)
                generated.write_text(f"Snow poles\n{condition} detail\nConclusion: Norway")
    return data, runs


def fill_caches(data: Path, runs: Path, skip: set[tuple[str, str]]) -> None:
    """Fake every cache the report reads, except the (pass, set) pairs in `skip`."""
    rounds = load_rounds(data)
    repo = CachedJudge(FunctionJudge(fake_reply), runs / "judge_cache" / "repo_s0.jsonl")
    for name in (*RELEASED, *GENERATED):
        _, pairs = load_set(name, data, runs)
        point_scores([(c.text, p.chain) for c, p in pairs], repo, "repo")
        folder = runs / "scores" / f"{name}__repo_s0"
        folder.mkdir(parents=True)
        with (folder / "rounds.csv").open("w") as f:
            f.write("round,truth,guess,correct,match\n")
            f.writelines(f"{r.key},{r.country},kg,{int(r.country == 'kg')},1\n" for r in rounds)
    for pass_name in PASSES:
        for name in [EXPERTS] if pass_name == "experts" else (*RELEASED, *GENERATED, EXPERTS):
            if (pass_name, name) not in skip:
                judge = CachedJudge(FunctionJudge(fake_reply), cache_path(runs, pass_name, name))
                run_pass(pass_name, name, data, runs, judge)


def n_values(by_round: dict[str, list[float]]) -> int:
    return sum(len(values) for values in by_round.values())


SKIP = {("pairwise", "A1"), ("location", "qwen3"), ("grounded", EXPERTS)}


def test_collect_scores_every_set_and_leaves_out_what_is_not_cached(runs_dataset):
    data, runs = runs_dataset
    fill_caches(data, runs, SKIP)
    collected = collect(data, runs)
    pairs = {s: n_values(collected[s]["shared"]["F1"]) for s in ("qwen", EXPERTS)}
    assert pairs == {"qwen": 6, EXPERTS: 6}  # 2 shared rounds x 3 references; 6 paper pairs
    assert n_values(collected["qwen"]["all"]["F1"]) == 7
    assert n_values(collected[EXPERTS]["reverse"]["F1"]) == 6
    assert value(collected, EXPERTS, "shared", "win_rate") == 50.0
    assert value(collected, "A1", "all", "o2o_F1") is None
    assert value(collected, "G0", "all", "o2o_F1") is not None
    assert value(collected, "qwen3", "shared", "median_km") is None
    assert value(collected, EXPERTS, "shared", "grounded_rate") is None
    assert value(collected, EXPERTS, "all", "median_km") is not None  # 7 expert chains
    assert value(collected, "qwen", "all", "gated_F1") is not None
    reverse = collected[EXPERTS]["reverse"]
    assert value(collected, EXPERTS, "shared", "F1") == pytest.approx(
        value({"e": {"r": reverse}}, "e", "r", "F1")
    )  # one-to-all F1 does not depend on the direction


def test_collect_with_no_caches_at_all_gives_no_metrics_and_no_error(runs_dataset):
    data, runs = runs_dataset
    collected = collect(data, runs)
    assert value(collected, "qwen", "all", "F1") is None
    assert value(collected, EXPERTS, "shared", "F1") is None


def run_report(data: Path, runs: Path, out: Path) -> subprocess.CompletedProcess:
    script = REPO / "scripts" / "metrics_report.py"
    cmd = [sys.executable, str(script), "--data", str(data), "--runs", str(runs), "--out", str(out)]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)


def test_report_end_to_end_on_fake_caches_prints_dashes_for_missing_ones(runs_dataset, tmp_path):
    data, runs = runs_dataset
    fill_caches(data, runs, SKIP)
    result = run_report(data, runs, tmp_path / "metrics.csv")
    assert result.returncode == 0, result.stderr
    out = result.stdout
    for title in ("Soft one-to-all F1", "Matching", "Specificity", "Location", "Summary"):
        assert title in out
    a1_matching = [line for line in out.splitlines() if line.startswith("A1 ")]
    assert any(" - " in line for line in a1_matching)
    with (tmp_path / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    keys = {(r["section"], r["set"], r["scope"], r["metric"]) for r in rows}
    assert {("metric", "qwen", "all", f"hard_F1_t{t}") for t in range(30, 100, 10)} <= keys
    assert ("metric", "A1", "all", "o2o_F1") not in keys
    assert ("summary", "E", "shared", "F1") in keys


def test_report_runs_before_any_cache_exists(runs_dataset, tmp_path):
    result = run_report(*runs_dataset, tmp_path / "metrics.csv")
    assert result.returncode == 0, result.stderr
    assert "Summary" in result.stdout
