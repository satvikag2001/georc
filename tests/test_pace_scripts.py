"""Guards for the PACE job scripts, run against a fake `sbatch` (no cluster needed)."""

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def run_bash(script: str, tmp_path: Path) -> str:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    sbatch = fake_bin / "sbatch"
    sbatch.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    sbatch.chmod(0o755)
    env = {
        "HOME": str(tmp_path),
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "GEORC_ROOT": str(tmp_path / "georc"),
    }
    out = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, check=True, cwd=REPO
    )
    return out.stdout


def submitted_args(tmp_path: Path, *args: str) -> list[str]:
    return run_bash(f"bash pace/submit.sh {' '.join(args)}", tmp_path).split("\n")[:-1]


def test_submit_uses_free_tier_and_bf16_gpus_only(tmp_path):
    args = submitted_args(tmp_path, "preflight.sbatch")
    assert "--account=gts-awu36" in args
    assert "--qos=embers" in args
    # A partition list alone let an embers job land on a V100 (no bf16); node features cannot.
    assert "--constraint=A100|H100|H200|L40S|RTX-Pro-Blackwell" in args
    assert "--job-name=georc-preflight" in args


def test_submit_after_adds_a_real_dependency_that_dies_with_its_parent(tmp_path):
    args = submitted_args(tmp_path, "--after", "123", "score.sbatch", "--seed", "1")
    assert "--dependency=afterok:123" in args
    assert "--kill-on-invalid-dep=yes" in args
    script = next(i for i, a in enumerate(args) if a.endswith("pace/score.sbatch"))
    assert args[script + 1 :] == ["--seed", "1"]  # script arguments stay after the script


def test_submit_without_after_sets_no_dependency(tmp_path):
    args = submitted_args(tmp_path, "score.sbatch")
    assert not any(a.startswith("--dependency") for a in args)


def test_common_sh_disables_jit_and_multiprocess_vllm_and_finds_venv_tools(tmp_path):
    out = run_bash(
        "source pace/common.sh; echo $VLLM_USE_FLASHINFER_SAMPLER $VLLM_ENABLE_V1_MULTIPROCESSING;"
        ' echo "${PATH%%:*}"',
        tmp_path,
    )
    flags, first_path = out.strip().split("\n")
    assert flags == "0 0"
    assert first_path == str(tmp_path / "georc" / "venv" / "bin")


def test_inside_a_job_every_cache_and_home_is_under_the_project_root(tmp_path):
    # Jobs 15386598-601 left ~/.cache/flashinfer and ~/.tilelang behind (libraries default to ~).
    out = run_bash(
        "SLURM_JOB_ID=1; source pace/common.sh;"
        " echo $HOME $FLASHINFER_WORKSPACE_BASE $TILELANG_CACHE_DIR $TILELANG_TMP_DIR",
        tmp_path,
    )
    root = str(tmp_path / "georc")
    assert all(path.startswith(root + "/") for path in out.split()), out


def test_outside_a_job_home_is_left_alone(tmp_path):
    assert run_bash("source pace/common.sh; echo $HOME", tmp_path).strip() == str(tmp_path)


def pilot_calls(tmp_path: Path, *args: str) -> list[list[str]]:
    """Run pace/pilot.sbatch with a stub common.sh whose python only echoes its arguments."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    (fake_bin / "python").write_text('#!/usr/bin/env bash\necho "$@"\n')
    (fake_bin / "nvidia-smi").write_text("#!/usr/bin/env bash\n")
    for tool in fake_bin.iterdir():
        tool.chmod(0o755)
    common = tmp_path / "scratch" / "georc" / "code" / "pace" / "common.sh"
    common.parent.mkdir(parents=True)
    common.write_text(f"PY={fake_bin / 'python'} GEORC_CODE=/c GEORC_DATA=/d GEORC_OUT=/o\n")
    return [
        line.split()
        for line in run_bash(f"bash pace/pilot.sbatch {' '.join(args)}", tmp_path).splitlines()
    ]


def test_pilot_generates_each_condition_then_scores_them_all_in_one_run(tmp_path):
    calls = pilot_calls(tmp_path)
    *generate, score = calls
    common = ["--data", "/d", "--out", "/o"]
    assert generate == [
        ["/c/scripts/generate_chains.py", *common, "--condition", c, "--seeds", "0", "--pilot"]
        for c in ["G0", "G0R", "A1"]
    ]
    assert score[: len(common) + 1] == ["/c/scripts/score_chains.py", *common]
    assert score[score.index("--variant") + 1] == "paper"
    assert {"--pilot", "--seed"} <= set(score) and score[score.index("--seed") + 1] == "0"
    dirs = score[score.index("--dirs") + 1 :]
    assert dirs == [f"{c}=/o/chains/{c}/seed0" for c in ["G0", "G0R", "A1"]]


def test_pilot_passes_the_judge_variant_and_takes_conditions_from_arguments(tmp_path):
    *generate, score = pilot_calls(tmp_path, "--variant", "repo", "A1")
    assert [g[g.index("--condition") + 1] for g in generate] == ["A1"]
    assert score[score.index("--variant") + 1] == "repo"
    assert score[score.index("--dirs") + 1 :] == ["A1=/o/chains/A1/seed0"]


def test_pilot_rejects_an_unknown_variant_before_spending_gpu_time_on_generation(tmp_path):
    with pytest.raises(subprocess.CalledProcessError) as failure:
        pilot_calls(tmp_path, "--variant", "fig21", "G0")
    assert "generate_chains.py" not in failure.value.stdout


def test_no_script_relies_on_sbatch_dependency_env_var():
    # sbatch has no SBATCH_DEPENDENCY input variable; relying on it silently drops the dependency.
    for path in (REPO / "pace").iterdir():
        assert "SBATCH_DEPENDENCY" not in path.read_text(), path.name


def authors_sbatch(tmp_path: Path, *args: str, patched: bool = True) -> subprocess.CompletedProcess:
    """Run pace/authors.sbatch with a stub common.sh whose python echoes its arguments."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    (fake_bin / "python").write_text('#!/usr/bin/env bash\necho "offline=$HF_HUB_OFFLINE" "$@"\n')
    (fake_bin / "nvidia-smi").write_text("#!/usr/bin/env bash\n")
    for tool in fake_bin.iterdir():
        tool.chmod(0o755)
    root = tmp_path / "scratch" / "georc"
    common = root / "code" / "pace" / "common.sh"
    common.parent.mkdir(parents=True)
    common.write_text(
        f"PY={fake_bin / 'python'} GEORC_ROOT={root} GEORC_CODE=/c GEORC_DATA=/d GEORC_OUT=/o"
        " HF_HOME=/h\n"
    )
    if patched:
        stub = root / "authors" / "GeoRC" / "src" / "vllm" / "vllm_completion_util.py"
        stub.parent.mkdir(parents=True)
        stub.touch()
    env = {"HOME": str(tmp_path), "PATH": f"{fake_bin}:/usr/bin:/bin"}
    cmd = ["bash", "pace/authors.sbatch", *args]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=REPO)


def test_authors_sbatch_runs_the_driver_offline_on_the_patched_clone(tmp_path):
    out = authors_sbatch(tmp_path, "--tags", "qwen3", "--shard", "1/2")
    assert out.returncode == 0, out.stderr
    [call] = out.stdout.split("\n")[:-1]
    root = tmp_path / "scratch" / "georc"
    assert call.split() == [
        "offline=1",
        "/c/scripts/authors_score.py",
        *("--authors", f"{root}/authors/GeoRC", "--data", "/d", "--out", "/o/authors"),
        *("--cache-dir", "/h/hub", "--tags", "qwen3", "--shard", "1/2"),
    ]


def test_authors_sbatch_stops_before_python_without_the_patched_clone(tmp_path):
    out = authors_sbatch(tmp_path, patched=False)
    assert out.returncode == 1
    assert "fetch_authors.sh" in out.stderr and "authors_score.py" not in out.stdout


@pytest.mark.skipif(
    not os.environ.get("GEORC_AUTHORS_REPO"), reason="set GEORC_AUTHORS_REPO to a local clone"
)
def test_fetch_authors_clones_the_pinned_commit_and_reapplies_the_patch_idempotently(tmp_path):
    root = tmp_path / "georc"
    (root / "venv" / "bin").mkdir(parents=True)
    (root / "venv" / "bin" / "python").write_text(
        "#!/usr/bin/env bash\ncat >/dev/null\necho checked\n"
    )
    (root / "venv" / "bin" / "python").chmod(0o755)
    (root / "code").symlink_to(REPO)
    script = f"GEORC_AUTHORS_URL={os.environ['GEORC_AUTHORS_REPO']} bash pace/fetch_authors.sh"
    dest = root / "authors" / "GeoRC"
    first = run_bash(script, tmp_path)
    assert "checked" in first
    score = dest / "score.py"
    patched = score.read_text()
    assert patched.count("REPRO PATCH") == 4
    score.write_text(patched + "# local drift\n")
    (dest / "stray.py").write_text("")
    run_bash(script, tmp_path)  # second run: reset to the commit, patch applied exactly once
    assert score.read_text() == patched and not (dest / "stray.py").exists()
    head = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    assert head.stdout.strip() == "a7a167979b3b355c29d59d36e3bff66c79f6a530"
