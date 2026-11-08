# Spec: Data-driven Gaussian initialization for 3D Gaussian Point Encoders

Status: Parked 2026-11-08 in favor of the GeoRC study (see ../SPEC.md). Draft; never implemented.

## Objective

**Question.** Does placing the Gaussian means where the data is, instead of uniformly at random, make
end-to-end training of the PointNet-style 3D Gaussian Point Encoder (3DGPE) stable?

**Source.** J. James, B. Wilson, S. Lucey, J. Hays, *3D Gaussian Point Encoders*, WACV 2026
(arXiv:2511.04797). Code: github.com/jimtjames/3dGaussianPointEncoders, commit `3e08aa7` (2026-10-02).

**Why this is worth testing.**
- The authors attribute the high variance of plain-optimizer training partly to "heightened sensitivity
  to parameter initialization, especially with respect to the Gaussian means" (Sec. 4.2.2).
- Table 2 (ScanObjectNN, validation, class-averaged accuracy, 10 trials): at 32 Gaussians, Adam scores
  52.4 ± 23.6, Mahalanobis natural gradient 78.2 ± 4.3, distillation 81.6 ± 4.9.
- Table 3: with distillation at 32 Gaussians, freezing the means at their initial values costs 19.4
  points (81.6 to 62.2). Where the means start clearly matters.
- The paper lists "requires more careful optimization techniques than PointNet" as a limitation (Sec. 5.1).
  If a better init lets plain Adam train reliably, that limitation shrinks.
- Cheap to run: the model has 1.39M parameters, and ScanObjectNN PB_T50_RS has 11,416 training scans
  (8,562 after the 25% validation hold-out).

**What the code actually does today** (read at `3e08aa7`; this changes the framing of the brief):
- `PointNet/encoder.py` supports `init_mode` = `random` (default), `uniform`, `layered`. None of them
  looks at the data. Every README command uses `random`.
- `random` draws means uniformly from `cloud_range`, which `train.py` sets to **[-5, 5]^3 for
  ScanObjectNN**. If the scans sit near the unit cube, most Gaussians start far from every point.
  Precision starts near identity (Cholesky diagonal 1, off-diagonal 0.01).
- The model has **two** Gaussian encoders: one inside the T-Net (sees raw points) and the main one
  (sees points multiplied by the T-Net's predicted 3x3 transform). Both use the same init.
- `train.py` never sets a seed. `distill.py` does (`set_seed`).
- The paper states the natural-gradient learning rate as 0.005; the `train.py` default is 1e-2.
- The README's "Fisher" example passes `--hook mahalanobis` (copy-paste error; Fisher is `--hook FIM`).
- No PointNet teacher checkpoint is included.
- The repository has no LICENSE file.

**Hypothesis.** k-means placement of the means, in the coordinate space each encoder sees, reduces
seed-to-seed spread under plain Adam, and may let 24 Gaussians match the accuracy that random init
needs 32 for.

**Who it is for.** Satvik runs the study. The output is a short report that either supports the idea
(worth writing up or sending to the authors) or is a clean negative result. A negative result is an
acceptable outcome; the authors may already have tried this.

**In scope.** PointNet-style 3DGPE, ScanObjectNN PB_T50_RS (`objectdataset_augmentedrot_scale75`),
optimizers Adam and Mahalanobis natural gradient, 16/24/32 Gaussians, 10 seeds per cell.

**Out of scope.** ModelNet40, Mamba3D, Fisher natural gradient, SOAP/Shampoo, any change to the
architecture or loss, covariance initialization (see Open Questions).

## Assumptions (confirm or correct)

1. "Natural gradient" means the Mahalanobis variant (`--dual_optim --hook mahalanobis`), which the paper
   uses for its headline results.
2. Only the means change between init methods. Covariances keep the repository default in every
   variant, so the comparison isolates mean placement.
3. **T-Net alignment.** Each encoder's means are fitted in the space that encoder sees at step 0:
   - T-Net encoder: raw training points.
   - Main encoder: training points multiplied by the transform from the freshly initialized T-Net
     (computed on a deep copy in training mode, so BatchNorm running statistics are untouched).
   The T-Net still drifts during training; that is part of what is being tested.
4. "Weighted by PointNet activations" means each point is weighted by its PointNet *critical-point
   count*: the number of the 1024 feature channels for which that point wins the max-pool. The T-Net
   encoder uses the teacher T-Net's activations and the main encoder uses the teacher encoder's.
   The two k-means variants differ only in these weights.
5. One PointNet teacher (seed 0) is shared by every `kmeans_pn` run.
6. The primary metric is best-epoch validation class-averaged accuracy (`val_macc_best`). That is what
   `train.py` uses to pick `best.pth`, and it is the only metric comparable to Table 2.
7. Test accuracy is always reported as a secondary metric. The validation split is a random 25% drawn
   per sample, and PB_T50_RS holds several perturbed copies of each object, so validation likely
   overstates generalization. The same model scores 78.2 on validation (Table 2) and 65.5 on test (Table 1).
8. Hyperparameters are `train.py` defaults (20 epochs, batch 32, `cls_lr` 1e-4, `weighting_lr` 1e-3,
   Adam means/covariance lr 1e-2). The one exception is the natural-gradient learning rate, which
   the pilot (P1) chooses.
9. Training runs on this machine (RTX 5070 Ti Laptop, 12 GB). PACE is used for latency only.
10. Class-averaged accuracy uses upstream's per-batch averaging (the yanx27 convention). We keep it
    as-is so results stay comparable to the paper.

## Experiment design

### Init variants (`--mean_init`)

| Id | Means | Needs |
|---|---|---|
| `repo` | Upstream `init_mode=random`: uniform in `cloud_range` ([-5, 5]^3) | nothing |
| `kmeans` | k-means centroids (k = number of Gaussians) of training points, fitted per encoder (Assumption 3) | training split |
| `kmeans_pn` | As `kmeans`, with each point weighted by its PointNet critical-point count (Assumption 4) | training split, teacher |

k-means settings: scikit-learn `KMeans`, `n_init=4`, `random_state=seed`, fitted on 1,024 training
scans sampled with the run seed (2,048 points each, no augmentation). The k-means randomness counts
toward seed variance, as it should. Validation and test points are never used.

### Phases and run counts

| Phase | Content | Runs |
|---|---|---|
| P0 preflight | Data check, point extent, dead-Gaussian fraction per init, timing of one epoch | 1 short |
| P1 pilot | Natural-gradient lr 1e-2 vs 5e-3; `repo`, Mahalanobis, 32 Gaussians, seeds 0-2 | 6 |
| P2 baseline (gate G1) | `repo` x {adam, mahalanobis} x {16, 24, 32} x seeds 0-9 | 60 |
| P3 teacher (gate G2) | PointNet without feature transform, ScanObjectNN training split, seed 0 | 1 |
| P4 data-driven init | {`kmeans`, `kmeans_pn`} x {adam, mahalanobis} x {16, 24, 32} x seeds 0-9 | 120 |
| P5 distillation reference (optional) | `distill.py`, 32 Gaussians, seeds 0-9 | 10 |
| P6 latency (PACE) | Every P2/P4 `best.pth`, GPU and CPU, unfiltered and bbox-filtered (t = 0.10) | 2 jobs |
| P7 analysis | Tables, criteria verdicts, `results/REPORT.md` | - |

P1 runs that use the chosen lr have the same run id as their P2 counterparts, so the sweep reuses
them. `kmeans` runs in P4 can start before the teacher is ready.

### Metrics recorded per run (`results.json`)

`val_macc_best`, `val_oa_best`, `best_epoch`, per-epoch validation curves, `test_macc` and `test_oa`
(of the best-validation checkpoint), dead-Gaussian fraction at init for both encoders, wall time, peak
GPU memory, and the full config: seed, init, optimizer, number of Gaussians, learning rates, upstream
commit, patch hash, torch version, GPU name.

A Gaussian is **dead at init** when its activation stays below 1e-3 on every point of 1,024 sampled
training scans. A run has **collapsed** when `val_macc_best` < 50.0.

### Latency protocol (P6)

- Batch size 1, 2,048 points, fp32, eval mode, 50 warm-up and 500 timed iterations. Report the median
  and p90 in ms. GPU timing uses CUDA events with synchronization.
- CPU: `torch.set_num_threads(1)`, plus a second pass using all allocated cores.
- Unfiltered latency depends only on the number of Gaussians. Bbox-filtered latency (`bbox_prune(0.10)`
  on both encoders, the paper's filtered setting) depends on where the Gaussians end up, so it is
  measured per checkpoint.
- Also recorded locally and independent of hardware: Gaussian-point pairs evaluated per sample, and
  validation class-averaged accuracy after filtering.
- Absolute numbers will not match Table 1. The paper used different hardware and applies the inverse
  T-Net transform to the Gaussians, which this code does not. Only relative comparisons are claimed.

## Tech Stack

- Python 3.12, PyTorch with CUDA (cu130 wheels, as in upstream `uv.lock`; Blackwell needs cu128 or later).
- Upstream dependencies as in `PointNet/pyproject.toml` (numpy, h5py, scipy, torchvision, mlflow, ray,
  torch-optimizer, pandas, ...). MLflow stays off.
- New: `scikit-learn` (k-means with `sample_weight`). Dev: `pytest`, `ruff`.
- Environment and lockfile: `uv`. Nix is not used (not installed here).
- PACE Phoenix (Slurm) for latency: account `gts-awu36`, QoS `embers`, A100 GPU (L40S via
  `-p gpu-l40s --gres=gpu:l40s:1` if A100 queues are long; never mix types within one measurement).
  Account, user and paths follow `~/director_project/v3/pace/common.sh`.

## Commands

```bash
# Setup (local)
bash scripts/setup_upstream.sh      # clone upstream @3e08aa7 into third_party/3dgpe, apply patches/, link data
uv sync                             # .venv: Python 3.12, torch, upstream deps, scikit-learn, pytest, ruff

# Quality
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
uv run pytest -q                    # CPU unit tests, under a minute
uv run pytest -q -m gpu             # GPU smoke test: 1-epoch train.py on synthetic data, all init variants

# Phases (sweep.py is resumable: it skips any run whose results.json exists)
uv run python scripts/preflight.py --ng 16 24 32 --out results/preflight.json
uv run python scripts/sweep.py --phase pilot    --parallel 2
uv run python scripts/sweep.py --phase baseline --parallel 2
uv run python scripts/train_teacher.py --seed 0 --epochs 200 --out runs/teacher/pointnet_noft.pth
uv run python scripts/sweep.py --phase init     --parallel 2 --teacher runs/teacher/pointnet_noft.pth
uv run python scripts/sweep.py --phase distill  --parallel 1 --teacher runs/teacher/pointnet_noft.pth
uv run python scripts/analyze.py --runs runs --out results

# Single run, for debugging (train.py uses paths relative to PointNet/)
cd third_party/3dgpe/PointNet && ../../../.venv/bin/python train.py --n 32 --batch_size 32 \
  --dataset scanobjnn_hard --dual_optim --hook mahalanobis --mean_lr 5e-3 --cov_lr 5e-3 \
  --epochs 20 --seed 0 --mean_init kmeans --run_dir ../../../runs/debug

# Latency on PACE (GT VPN required). Same layout as ~/director_project/v3/pace:
# code in ~/scratch/gpe-init, artifacts in a sibling dir so an rsync --delete can never reach them.
rsync -a --exclude .venv --exclude data --exclude runs ./ \
  sagrawal394@login-phoenix.pace.gatech.edu:scratch/gpe-init/
rsync -a --include '*/' --include 'best.pth' --include 'results.json' --exclude '*' runs/ \
  sagrawal394@login-phoenix.pace.gatech.edu:scratch/gpe-init_artifacts/runs/
#   on PACE: `pace-quota` must list gts-awu36; then `uv python install 3.12 && uv sync`
#   (the cluster python module is 3.10, upstream needs 3.12)
sbatch pace/latency_gpu.sbatch
sbatch pace/latency_cpu.sbatch
rsync -a sagrawal394@login-phoenix.pace.gatech.edu:scratch/gpe-init_artifacts/results/latency_*.csv results/
```

Header shared by both sbatch files (GPU type fixed for every measurement and recorded):

```bash
#SBATCH --account=gts-awu36          # free tier only; never gts-awu36-paid
#SBATCH --qos=embers                 # no charge; never inferno (inferno draws on paid credits)
#SBATCH --nodes=1 --ntasks-per-node=8 --gres=gpu:A100:1   # CPU job: drop --gres
#SBATCH --exclusive                  # reduce timing noise from co-located jobs, if embers allows it
#SBATCH --time=8:00:00               # director_project rule: wallclock is always 8h (a cap, not a reservation)
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sagrawal394@gatech.edu
#SBATCH --output=%x-%j.out
```

## Project Structure

```
CV_project_trial/
├── SPEC.md                    this file
├── pyproject.toml, uv.lock    environment + the gpe_init package
├── src/gpe_init/
│   ├── seeding.py             set_seed(): random, numpy, torch, cuda
│   ├── init_strategies.py     repo / kmeans / kmeans_pn; apply_mean_init(model, train_points, ...)
│   ├── teacher.py             load teacher; critical-point weights
│   ├── grid.py                single source of truth: phase -> list of run configs and run ids
│   ├── results.py             results.json schema, read/write
│   ├── stats.py               std ratio, bootstrap CI, Brown-Forsythe, Welch CI
│   └── criteria.py            evaluates G1, G2, S1, S2 from results
├── scripts/                   setup_upstream.sh, preflight.py, sweep.py, train_teacher.py,
│                              latency.py, analyze.py
├── patches/                   git format-patch series against upstream (train.py only)
├── third_party/3dgpe/         upstream clone + patches (gitignored; recreated by setup script)
├── pace/                      latency_gpu.sbatch, latency_cpu.sbatch, latency.sh
├── tests/                     test_*.py; conftest builds a tiny synthetic ScanObjectNN-format h5
├── data/ScanObjectNN/main_split/   gitignored; *_objectdataset_augmentedrot_scale75.h5
├── runs/                      gitignored; one directory per run id
└── results/                   committed: preflight.json, summary.csv, latency_*.csv, REPORT.md
```

**Upstream patch scope** (`train.py` only): add `--seed`, `--mean_init`, `--teacher_checkpoint` and
`--run_dir`; call `gpe_init.apply_mean_init(...)` right after the model is built (before hooks and
optimizers); write `results.json`. All other logic lives in `src/gpe_init`.

## Code Style

Python 3.12, type hints on public functions, `ruff format` (line length 100), snake_case. Docstrings
state tensor shapes the way upstream does (`(B, N, 3)`). Functions are pure and take an explicit seed;
there is no module-level state. Errors fail loudly with the offending shape or value.

```python
def kmeans_means(
    points: np.ndarray,
    n_gaussians: int,
    seed: int,
    weights: np.ndarray | None = None,
) -> torch.Tensor:
    """Cluster training points and return the centroids as Gaussian means.

    Args:
        points: (P, 3) float32, in the coordinate space the encoder sees.
        weights: (P,) non-negative per-point weights; None means uniform.

    Returns:
        (G, 3) float32 centroids.
    """
    if weights is not None and not np.any(weights > 0):
        raise ValueError("all k-means weights are zero")
    km = KMeans(n_clusters=n_gaussians, n_init=4, random_state=seed)
    km.fit(points, sample_weight=weights)
    return torch.from_numpy(km.cluster_centers_).float()


@torch.no_grad()
def set_means(encoder: GaussianEncoder, means: torch.Tensor) -> None:
    """Overwrite an encoder's means in place; covariances keep the repository default."""
    if means.shape != encoder.mu.shape:
        raise ValueError(f"expected {tuple(encoder.mu.shape)}, got {tuple(means.shape)}")
    encoder.mu.copy_(means.to(encoder.mu.device))
```

## Testing Strategy

pytest. Unit tests run on CPU in under a minute; GPU tests carry `@pytest.mark.gpu`. Every public
function in `src/gpe_init` has at least one test. `stats.py` and `criteria.py` get full branch
coverage, because they decide the verdict.

- **init_strategies:** recovers 4 planted, well-separated clusters (centroid error < 0.05); same seed
  gives identical means; different seeds give different means; weights pull centroids toward heavy
  points; all-zero weights raise an error. With the T-Net forced to a known rotation, the main
  encoder's means equal the rotated raw-space centroids (alignment). Only `mu` changes in the state dict:
  covariances, BatchNorm running statistics and the Parameter objects held by the optimizer stay the same.
- **seeding:** two models built with the same seed have bit-identical initial `mu` for every variant.
  Final accuracy is not required to match, since some CUDA kernels are nondeterministic.
- **teacher:** critical-point counts on a hand-built activation tensor equal per-channel argmax
  counts, and each scan's counts sum to 1024.
- **grid / sweep:** `init` phase yields exactly 120 unique run ids and `baseline` 60; a completed run is
  skipped and never overwritten.
- **stats / criteria:** fixed fixtures with hand-computed std ratio, bootstrap CI, Brown-Forsythe p and
  Welch CI; G1/G2/S1/S2 pass/fail on synthetic results built to sit on each side of every threshold.
- **latency:** timer returns sane median/p90 for a dummy module; bbox-filtered encoder output matches
  unfiltered output within 1e-3 at a tiny threshold.
- **GPU smoke:** a 1-epoch `train.py` run on the synthetic h5 writes a valid `results.json` for each
  of `repo`, `kmeans` and `kmeans_pn`.

Experiment-level correctness is enforced by the gates below, not by unit tests.

## Boundaries

**Always**
- Run `ruff` and `pytest -q` before committing and before launching any phase.
- Seed every run. Record the full config, upstream commit, patch hash and hardware in `results.json`.
- Fit k-means and compute PointNet weights on the 75% training split only.
- Report every run in the grid, including collapsed and crashed runs. Never drop seeds after the fact.
- Evaluate criteria exactly as written in this spec.
- On PACE use account `gts-awu36` with QoS `embers`, and check `pace-quota` before the first submission.
- Keep commit messages, PR descriptions and repository descriptions under 50 words, with no
  AI-assistant attribution of any kind (no tool names, no co-author trailers).

**Ask first**
- Changing any threshold, seed count, Gaussian count, hyperparameter or init definition once P2 has started.
- Adding variants (for example the controls in Open Questions) or dependencies beyond those listed.
- Submitting PACE jobs, or moving training to PACE.
- Editing upstream files other than `train.py`.
- Downloading datasets (ScanObjectNN is distributed under terms of use).
- `git init`, committing, pushing, or contacting the authors.
- Deleting anything under `runs/` or `results/`.
- Requesting a PACE container (a process that takes months).

**Never**
- Use `gts-awu36-paid`, the `inferno` QoS, any other PACE account, or any paid compute, credits or service.
- Select checkpoints or tune anything on the test split.
- Publish the vendored upstream code (no license) or the ScanObjectNN data.
- Skip, delete or weaken failing tests, or swallow errors, to get a sweep through.
- Present runs outside the grid as part of the main result without labelling them.

## Success Criteria

### Gates (must pass before the next phase)

- **G0 preflight:** the data loads with the expected split sizes (train 8,562, validation 2,854,
  test 2,882). `results/preflight.json` reports the per-axis point extent (min, max, 1st and 99th
  percentiles), the dead-Gaussian fraction for `repo` and `kmeans` at 16/24/32 Gaussians, peak GPU
  memory, and projected wall time for P1 to P5. If the projected time exceeds 48 hours, stop and ask.
- **G1 reproduction (after P2, at 32 Gaussians, `repo` init):** mean Mahalanobis `val_macc_best` falls
  in [73.9, 82.5] (Table 2's 78.2 ± 4.3), **and** Adam shows the instability: std at least 10.0, or at
  least 2 of 10 seeds collapsed. If Mahalanobis misses, stop and debug hyperparameters. If Adam is
  stable, stop and report: the problem this study targets did not reproduce.
- **G2 teacher:** test overall accuracy at least 66.0 (the paper's PointNet without feature transform
  reaches 69.3), trained on the training split only.

### Study verdict (pre-registered; each evaluated for `kmeans` and `kmeans_pn`)

- **S1 stability:** for Adam at 32 Gaussians, std of `val_macc_best` across 10 seeds is at most
  0.5 x the std of `repo` init (reproduced values, not the paper's), **and** the mean is no lower than
  the `repo` mean. The report includes the bootstrap 95% CI of the std ratio, the Brown-Forsythe
  p-value and the collapse count.
- **S2 efficiency:** at 24 Gaussians (either optimizer), mean `val_macc_best` is at least the reproduced
  `repo` Mahalanobis mean at 32 Gaussians minus 1.0, **and** the median PACE latency at 24 Gaussians is
  below that at 32, on both GPU and CPU. The report includes the Welch 95% CI of the difference. Adam's
  32-Gaussian result is not used as the bar, because it already falls below Adam's 24-Gaussian result
  in the paper.
- **Test-split direction check:** a criterion that passes on validation must show the same direction
  on `test_macc` (std ratio below 1 for S1, difference at least -1.0 for S2). Otherwise it is reported
  as "validation only".
- A strong result is S1 or S2 passing. Mahalanobis std ratios are reported as exploratory results.

### Done when

All grid runs are complete or their failures are documented, and the following are committed:
`results/summary.csv` (a Table 2-style mean ± std for every cell, on both validation and test),
`results/latency_*.csv`, and `results/REPORT.md` with the G0-G2 outcomes, the S1/S2 verdicts
(including negative ones) and the commands to regenerate every number.

## Open Questions

1. **PACE access (mostly resolved).** Account `gts-awu36`, user `sagrawal394` and host
   `login-phoenix.pace.gatech.edu` come from `~/director_project`. The one check left is that
   `pace-quota` still lists `gts-awu36` at submission time. If it does not, PACE work stops and no
   other account is substituted.
2. **Data (resolved).** Downloaded on 2026-11-06 from Hugging Face `cminst/ScanObjectNN`
   (`main_split/`) into `data/ScanObjectNN/main_split/`. Two other mirrors (`wangshengzhe/ScanObjectNN`,
   `adfishing/scanobjectnn-h5`) hold byte-identical files. Verified: train (11416, 2048, 3), test
   (2882, 2048, 3), 15 classes. SHA-256:
   - `training_objectdataset_augmentedrot_scale75.h5` `bd1ebda43155b112fac0ec9335723e66c4885449022002a2a8d370b56376eed5`
   - `test_objectdataset_augmentedrot_scale75.h5` `f23822ee2f1e1d18331e91334d74adb10c8718d72d503bfa76ebe0a9348a7ea1`
3. **Contact Jim James before P4?** (Recommended.) A short email could settle four unknowns at once:
   whether data-driven init was tried, the epochs and learning rates behind Table 2, whether Table 2
   reports best-epoch validation accuracy, and whether the per-sample validation split leaks perturbed
   copies of training objects.
4. **Bbox-random control (now recommended).** Measured 2026-11-06: 97.2% of training points lie inside
   [-1, 1]^3, and the 1st-99th percentiles are within +/-0.92 on every axis. `repo` init spreads means
   over [-5, 5]^3, so only about 0.8% of that volume overlaps the data, which is about 0.26 of 32
   Gaussians on average. `kmeans` could therefore win simply by putting Gaussians inside the data,
   with the clustering adding nothing. A `random_bbox` variant (uniform within the training points'
   1st-99th percentile box) separates the two effects. Cost: 60 runs.
5. **Covariance init.** The default covariance is roughly the identity (sigma near 1). If the scans
   span about +/-1, every Gaussian then covers most of the object, which could hide any benefit from
   better means. Setting precision from each k-means cluster's spread is the natural follow-up if P4
   shows no effect.
6. **Version control (resolved).** Git repository with remote `satvikag2001/CV_project_trial`
   (private, because the upstream code has no license). `data/`, `runs/` and `third_party/3dgpe/`
   are gitignored.
