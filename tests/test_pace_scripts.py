"""Guards for the PACE job scripts, run against a fake `sbatch` (no cluster needed)."""

import subprocess
from pathlib import Path

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
    assert "--constraint=A100|H100|H200|L40S" in args
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


def test_no_script_relies_on_sbatch_dependency_env_var():
    # sbatch has no SBATCH_DEPENDENCY input variable; relying on it silently drops the dependency.
    for path in (REPO / "pace").iterdir():
        assert "SBATCH_DEPENDENCY" not in path.read_text(), path.name
