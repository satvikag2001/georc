"""The harness around the authors' scorer; set GEORC_AUTHORS_REPO to a clone to run them all."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from georc import authors
from georc.data import chain_prefix
from georc.judge import parse_score

REPO = Path(__file__).resolve().parents[1]
AUTHORS_COMMIT = "a7a167979b3b355c29d59d36e3bff66c79f6a530"
PATCH = REPO / "patches" / "authors_a7a1679.patch"
AUTHORS_REPO = os.environ.get("GEORC_AUTHORS_REPO")
needs_source = pytest.mark.skipif(
    not AUTHORS_REPO, reason="set GEORC_AUTHORS_REPO to a clone of the authors' repository"
)
ROUND_TEXT = (
    "Round 1\n{0} one\nRound 2\n{0} two\nRound 3\n{0} three\nRound 4\n{0} four\nRound 5\n{0} five\n"
)


def make_dataset(root: Path, experts: dict[str, list[str]], tags=("qwen", "hallucinated")) -> Path:
    for challenge, names in experts.items():
        folder = root / challenge
        folder.mkdir(parents=True)
        meta = [{"streakLocationCode": "kg"} for _ in range(5)]
        (folder / f"{challenge}_rounds_metadata.json").write_text(json.dumps(meta))
        for name in names:
            (folder / f"{name}.txt").write_text(ROUND_TEXT.format(f"{name} saw road lines"))
        for tag in tags:
            for i in range(1, 6):
                chain = f"1. Cyrillic script on signs\n2. {tag} white bollards round {i}\n"
                (folder / f"{chain_prefix(tag)}{i}.txt").write_text(chain)
    return root


def test_units_cover_every_expert_of_every_challenge_and_shards_partition_them(tmp_path):
    data = make_dataset(
        tmp_path, {"b": ["Human_Expert_1"], "a": ["Human_Expert_3", "Human_Expert_1"]}
    )
    units = authors.list_units(data, ["qwen", "hallucinated"])
    assert [(u.tag, u.challenge, u.expert) for u in units[:3]] == [
        ("qwen", "a", "Human_Expert_1"),
        ("qwen", "a", "Human_Expert_3"),
        ("qwen", "b", "Human_Expert_1"),
    ]
    assert len(units) == 6
    shards = [authors.shard(units, f"{i}/4") for i in range(1, 5)]
    flat = [u for s in shards for u in s]
    assert len(flat) == len(units) and set(flat) == set(units)
    assert authors.shard(units, "1/1") == units
    with pytest.raises(ValueError):
        authors.shard(units, "0/2")


def test_view_links_the_expert_as_human_chain_3_and_can_be_rebuilt(tmp_path):
    data = make_dataset(tmp_path / "data", {"a": ["Human_Expert_2"]})
    unit = authors.Unit("hallucinated", "Human_Expert_2", "a")
    root = authors.build_view(data, tmp_path / "views", unit)
    assert authors.build_view(data, tmp_path / "views", unit) == root
    folder = root / "a"
    assert [p.name for p in root.iterdir()] == ["a"]  # their pipeline lists challenge folders
    assert (folder / "Human_Chain_3.txt").resolve() == (data / "a" / "Human_Expert_2.txt").resolve()
    assert sorted(p.name for p in folder.iterdir()) == [
        "Human_Chain_3.txt",
        *(f"hallucinated_reasoning_chain_{i}.txt" for i in range(1, 6)),
    ]


def test_view_fails_on_a_missing_candidate_where_their_pipeline_would_skip_it(tmp_path):
    data = make_dataset(tmp_path / "data", {"a": ["Human_Expert_1"]})
    (data / "a" / "candidate_reasoning_chain_qwen_4.txt").unlink()
    with pytest.raises(FileNotFoundError):
        authors.build_view(data, tmp_path / "views", authors.Unit("qwen", "Human_Expert_1", "a"))


@pytest.mark.parametrize(
    "reply, metric, expected",
    [
        # The judge follows the prompt's only example, "precision", even for recall calls.
        ('{"precision": 80}', "recall", {"fallback": 80, "precision": 80, "released": 0}),
        ('{"recall": 40}', "recall", {"fallback": 40, "precision": 0, "released": 40}),
        ('```json\n{"precision": 70}\n```', "precision", dict.fromkeys(authors.SCORE_KEYS, 70)),
        ('{"precision": null}', "precision", dict.fromkeys(authors.SCORE_KEYS, 0)),
        ('{"precision": "85"}', "precision", dict.fromkeys(authors.SCORE_KEYS, 85)),
        ("I would say 60", "precision", dict.fromkeys(authors.SCORE_KEYS, 0)),
        ('{"precision": 150}', "precision", dict.fromkeys(authors.SCORE_KEYS, 150)),  # unclipped
    ],
)
def test_read_score_repeats_the_patched_parse_for_each_score_key(reply, metric, expected):
    for key, value in expected.items():
        assert authors.read_score(reply, metric, key) == value
    ours = parse_score(reply)
    assert authors.read_score(reply, metric, "first_number") == (0.0 if ours is None else ours)


def test_read_score_rejects_an_unknown_reading():
    with pytest.raises(ValueError):
        authors.read_score("no json here", "precision", "loose")


def test_pair_metrics_average_each_side_then_take_f1():
    calls = [
        {"metric": "precision", "reply": '{"precision": 60}'},
        {"metric": "precision", "reply": '{"precision": 20}'},
        {"metric": "recall", "reply": '{"precision": 40}'},
    ]
    assert authors.pair_metrics(calls, "fallback") == pytest.approx(
        {"precision": 40.0, "recall": 40.0, "f1": 40.0}
    )
    assert authors.pair_metrics(calls, "released")["f1"] == 0.0
    assert authors.pair_metrics([], "fallback") == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_summary_is_the_mean_of_per_pair_values_not_f1_of_means():
    pairs = [
        {"precision": 100.0, "recall": 0.0, "f1": 0.0},
        {"precision": 0.0, "recall": 100.0, "f1": 0.0},
    ]
    summary = authors.summarize(pairs)
    assert summary["f1"] == 0.0 and summary["f1_of_means"] == 50.0 and summary["n"] == 2
    low, high = authors.summarize([{"precision": 1, "recall": 1, "f1": f} for f in (10, 30)])[
        "f1_ci95"
    ]
    assert low < 20 < high and 20 - low == pytest.approx(high - 20)


def test_unit_seed_is_stable_and_differs_between_units():
    a, b = authors.Unit("qwen", "Human_Expert_1", "x"), authors.Unit("qwen", "Human_Expert_2", "x")
    assert authors.unit_seed(a, 0) == authors.unit_seed(a, 0) != authors.unit_seed(b, 0)
    assert 0 <= authors.unit_seed(a, 7) < 2**31


def test_written_units_load_back_as_pairs(tmp_path):
    unit = authors.Unit("qwen", "Human_Expert_1", "a")
    record = {
        "score_key": "fallback",
        "results": [{"challenge": "a", "round": 1, "precision": 1.0, "recall": 2.0, "f1": 1.3}],
        "calls": [[{"metric": "precision", "reply": "{}", "value": 0.0}]],
    }
    authors.write_json(unit.path(tmp_path), record)
    assert authors.completed_units(tmp_path, "qwen") == [unit]
    assert authors.completed_units(tmp_path, "gemma3_12b") == []
    [row] = authors.load_pairs(tmp_path, [unit])
    assert (
        row["expert"] == "Human_Expert_1"
        and row["round"] == 1
        and row["calls"] == record["calls"][0]
    )
    assert not list(unit.path(tmp_path).parent.glob(".*.tmp"))


def test_patch_touches_only_plumbing_and_removes_only_the_known_lines():
    text = PATCH.read_text()
    files = re.findall(r"^diff --git a/(\S+) b/", text, re.MULTILINE)
    assert files == [
        "score.py",
        "src/hugging_face/hf_completion_utils.py",
        "src/llm/llm_scoring.py",
        "src/vllm/vllm_completion_util.py",
    ]
    removed = [line[1:].strip() for line in text.splitlines() if re.match(r"^-(?!--)", line)]
    assert removed == [
        "from sentence_transformers import SentenceTransformer",
        "Please respond **only** with a JSON object. No explanation. Example format:"
        ' {"precision": <score>}.',
        "return float(data.get(metric_type, 0))",
        "except (ValueError, KeyError):",
        "scorer = ScorerFactory.get_scorer(",
        "model=llm,",
        "tokenizer=tokenizer,",
        "tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir,"
        ' device_map="cuda")',
        "model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir).eval()",
        "from sentence_transformers import util",
    ]
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", text)  # no Hugging Face token


def patched_clone(tmp_path: Path) -> Path:
    dest = tmp_path / "GeoRC"
    subprocess.run(["git", "clone", "-q", AUTHORS_REPO, str(dest)], check=True)
    git = ["git", "-C", str(dest)]
    subprocess.run([*git, "checkout", "-q", "--detach", AUTHORS_COMMIT], check=True)
    subprocess.run([*git, "apply", "--check", str(PATCH)], check=True)
    subprocess.run([*git, "apply", str(PATCH)], check=True)
    return dest


@needs_source
def test_patch_applies_cleanly_to_a7a1679(tmp_path):
    dest = patched_clone(tmp_path)
    assert (dest / "src" / "vllm" / "vllm_completion_util.py").exists()


STUBS = {  # imported by their modules at load time; never used with the fake judge
    "torch/__init__.py": "",
    "transformers/__init__.py": "def __getattr__(name):\n    return type(name, (), {})\n",
    "PIL/__init__.py": "",
    "PIL/Image.py": "",
}


def run_driver(tmp_path: Path, authors_dir: Path, data: Path, out: Path, *extra: str) -> str:
    stubs = tmp_path / "stubs"
    for name, body in STUBS.items():
        (stubs / name).parent.mkdir(parents=True, exist_ok=True)
        (stubs / name).write_text(body)
    cmd = [sys.executable, str(REPO / "scripts" / "authors_score.py"), "--fake-judge"]
    cmd += ["--authors", str(authors_dir), "--data", str(data), "--out", str(out), *extra]
    env = {**os.environ, "PYTHONPATH": str(stubs)}
    return subprocess.run(cmd, env=env, capture_output=True, text=True, check=True).stdout


@needs_source
def test_their_scorer_runs_end_to_end_with_a_stub_judge(tmp_path):
    authors_dir = patched_clone(tmp_path)
    data = make_dataset(
        tmp_path / "data", {"a": ["Human_Expert_1", "Human_Expert_3"], "b": ["Human_Expert_2"]}
    )
    out = tmp_path / "out"
    run_driver(tmp_path, authors_dir, data, out, "--tags", "qwen", "--limit", "2")
    run_dir = out / "fake"
    assert len(authors.completed_units(run_dir, "qwen")) == 2
    assert not (run_dir / "qwen" / "summary.json").exists()  # one unit still to do
    stdout = run_driver(tmp_path, authors_dir, data, out, "--tags", "qwen")
    assert "1 to do" in stdout

    rows = authors.load_pairs(run_dir, authors.completed_units(run_dir, "qwen"))
    assert len(rows) == 15
    for row in rows:
        expected = authors.pair_metrics(row["calls"], "fallback")
        assert {k: row[k] for k in expected} == pytest.approx(expected)
        metrics = [c["metric"] for c in row["calls"]]
        assert metrics == sorted(metrics)  # all precision calls, then all recall calls
        assert metrics.count("precision") == 2 and metrics.count("recall") == 1
    summary = json.loads((run_dir / "qwen" / "summary.json").read_text())
    assert summary["n"] == 15 and summary["score_key"] == "fallback"
    assert summary["f1"] == pytest.approx(authors.summarize(rows)["f1"])
    assert len(json.loads((run_dir / "qwen" / "pairs.json").read_text())) == 15


@needs_source
def test_released_score_key_reproduces_the_zero_recall_bug(tmp_path):
    authors_dir = patched_clone(tmp_path)
    data = make_dataset(tmp_path / "data", {"a": ["Human_Expert_1"]})
    out = tmp_path / "out"
    run_driver(
        tmp_path, authors_dir, data, out, "--tags", "hallucinated", "--score-key", "released"
    )
    rows = authors.load_pairs(out / "fake", authors.completed_units(out / "fake", "hallucinated"))
    assert len(rows) == 5 and all(r["recall"] == 0 and r["f1"] == 0 for r in rows)
    assert any(r["precision"] > 0 for r in rows)
