# Tasks: GeoRC Fix A and Fix B

Plan: `tasks/plan.md`. Each task ends with `uv run ruff check . && uv run pytest -q` passing locally.
Commits approved: commit after each task and push to `origin`; messages under 30 words, no AI attribution.
Scope (2026-11-10): Gemma-3-12B only, seed 0. Qwen conditions (Q0, Q-B0, Q-B1) are deferred.

PACE conventions for every task:
- Account `gts-awu36`, QoS `embers`, job names `georc-*`, wallclock 1-2h (short jobs backfill sooner).
- Everything under `~/scratch/georc/`, with `HF_HUB_OFFLINE=1` in jobs.

---

## Phase 0: Spec

### T1: GeoRC spec; park 3DGPE spec
**Description:** Write `SPEC.md` for the GeoRC study from `tasks/plan.md`, covering objective,
commands, structure, style, testing and boundaries, including the PACE and cleanup rules. Move the
current 3DGPE spec to `specs/3dgpe.md` with a "Parked 2026-11-08" status line.
**Acceptance:**
- [x] `SPEC.md` covers all six areas and states gates G1 and G2 and the win rule.
- [x] `specs/3dgpe.md` is identical to the old `SPEC.md` apart from the status line.
**Verify:** `git diff --stat`; human reads and approves `SPEC.md`.
**Depends on:** none. **Files:** `SPEC.md`, `specs/3dgpe.md`. **Size:** XS

---

## Phase 1: A trustworthy scorer

### T2: Dataset fetch, parsing, point extraction
**Description:** Create the `georc` package and a script that downloads the HF dataset with a token.
Parse the data:
- challenges, rounds and the ground-truth ISO country from `*_rounds_metadata.json`;
- expert chains from `Human_Expert_N.txt`, split by "Round N", following the paper's parse rules:
  drop lines containing "reasoning", stop at "conclusion";
- candidate chains, by model tag.

Point extraction follows the paper: split lines, strip `1.`, `*` or `-` prefixes, keep lines longer
than 5 characters.
**Acceptance:**
- [x] Loader yields exactly 500 rounds and 800 (round, expert) pairs. Released chains: 500 each
  verified locally for `gemma3_12b` and `qwen`; the other 10 sets are checked on PACE in T4.
- [x] Parsing matches the paper's rules on fixture files, including a line with "reasoning" in it
  and a "Conclusion" block.
- [x] Ground-truth country for challenge `1NJsXTxIF9GGMDxC` is `[kg, my, rs, ar, za]`.
- [x] Found: round `8Uo6ejwXYqmp9av3_4` has a null country code; filled with `kr` from its
  coordinates (documented override in `data.py`).
**Verify:** `uv run pytest -q tests/test_data.py`.
**Depends on:** none.
**Files:** `pyproject.toml`, `src/georc/data.py`, `src/georc/points.py`, `scripts/fetch_dataset.py`,
`tests/test_data.py`. **Size:** M

### T3: One-to-all scorer and country check, with a pluggable judge
**Description:** Judge interface with a `FakeJudge` for tests.
- **Prompt:** the judge prompt verbatim from paper Fig. 14, diffed against the repo string; the
  paper wins where they differ.
- **Parsing:** the score is the first number inside the first `{...}`. Missing or garbled replies
  score 0, and that count is recorded.
- **Per-pair metrics:** precision is each candidate point scored against the whole expert chain
  joined by `||`, averaged. Recall is each expert point scored against the whole candidate chain.
  F1 combines them.
- **Country check:** the judge extracts the predicted country as an ISO-2 code from the chain's
  conclusion, which is compared to ground truth.
- **Caching:** results are cached per judge call (keyed by prompt hash and seed), so reruns are free.
**Acceptance:**
- [x] A toy pair with hand-computed scores gives the expected P, R and F1.
- [x] Parser edge cases: a `recall` key, extra text, no JSON, out-of-range numbers (clipped to 0-100).
- [x] An empty candidate or empty expert chain gives 0/0/0, as in the paper.
**Verify:** `uv run pytest -q tests/test_score.py`.
**Depends on:** T2.
**Files:** `src/georc/judge.py`, `src/georc/score.py`, `src/georc/country.py`, `tests/test_score.py`.
**Size:** M

### T4: PACE environment, asset staging, preflight
**Description:**
- **`pace/common.sh`:** account and QoS pair; paths under `~/scratch/georc`; `HF_HOME`,
  `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `TMPDIR` and the pip cache on scratch; `HF_HUB_OFFLINE=1`
  in jobs; the `georc-` job prefix.
- **`pace/setup_env.sh`** (login node): install uv, Python 3.12, a pinned vLLM and this package.
- **`pace/fetch_assets.sh`** (login node, HF token): download the dataset and three models,
  Gemma-3-12B-it, Qwen2.5-VL-7B-Instruct and Qwen3-4B-Instruct-2507.
- **`pace/sync.sh`:** push code only, never outputs.
- **`pace/submit.sh`:** request `gpu:1` across a100,h100,h200,l40s.
- **`pace/preflight.sbatch`:** run `nvidia-smi`, load each model offline, generate one Gemma image
  with and without pan-and-scan and one Qwen image, and make one judge call.

Before any download, record home usage, scratch contents and the director job count to
`results/pace_baseline.txt`, for the cleanup check.
**Acceptance:**
- [x] The preflight job exits 0 on `embers`, and its log shows driver/CUDA, all 3 models loaded offline, and that pan-and-scan works (or the fallback is chosen). Job 15386598 on an H200: judge 100/0 on match/mismatch, Qwen prompt 5,907 tokens, Gemma pan-and-scan +539 prompt tokens.
- [x] Home usage has not grown by more than 0.5 GB (14.4 GB before and after, 2026-11-08).
- [x] Nothing exists outside `~/scratch/georc`: FlashInfer and TileLang caches found in home and removed
  2026-11-10; inside jobs, `HOME` and both caches now point at scratch (guard test added).
- [x] All 12 released chain sets have 500 chains each on PACE, plus all 500 images (carried over
  from T2).
- [x] Home footprint found and removed: uv's `~/.local/bin/python3.12` link and its
  `~/.config/uv/uv-receipt.json` receipt. `common.sh` and `setup_env.sh` now prevent both.
- Status 2026-11-08: env built (vLLM 0.32.0, torch 2.13 + CUDA 13.0), all assets downloaded.
  Preflight job 15325036 is queued on embers; the G1 jobs 15325037-38 wait on it (afterok).
- 2026-11-09: all three jobs failed. Causes and fixes (guarded by `tests/test_pace_scripts.py`):
  V100 placement despite the partition list (now a node-feature constraint); FlashInfer sampler
  JIT needing a compiler (PyTorch sampler); "device busy" on an H100 (single-process vLLM);
  `SBATCH_DEPENDENCY` does not exist (explicit `--after`); 35-min startup from uncompiled bytecode
  (venv precompiled). Resubmitted: preflight 15386598, G1 15386599 and 15386601 (afterok).
**Verify:** `ssh pace 'tail -50 ~/scratch/georc/logs/georc-preflight-*.out'`; `ssh pace 'pace-quota | grep Home'`.
**Depends on:** none (HF token and Gemma license, both provided).
**Files:** `pace/common.sh`, `pace/setup_env.sh`, `pace/fetch_assets.sh`, `pace/submit.sh` (+ `sync.sh`),
`pace/preflight.sbatch`. **Size:** M

### T5: Validate the scorer on released chains (gate G1)
**Description:** `scripts/score_chains.py` plus a vLLM judge backend, run by `pace/score.sbatch`.
Score the released chains of Qwen2.5-VL, Gemma-3, Llama-3.2 and Qwen3-VL, plus the hallucinated
baseline, with the judge's default sampling and seed 0. Gemma was repeated with judge seed 1 to
measure judge noise (done: 0.02 F1). Write a validation CSV comparing ours with Table 2, with both
country measures (strict `country_acc` and `country_acc_match`).
**Acceptance (G1):**
- [ ] Each open model's F1 is within its paper 95% CI +/- 1.0, and the ranking holds.
- [ ] Country accuracy is within +/- 3 points of Table 2. The hallucinated baseline's F1 is within +/- 2 of 18.13.
- [ ] The F1 difference between the two judge seeds is reported. If G1 fails, try greedy decoding once, then the released-code rubric (`variant="repo"`); if both fail, stop and report.
**Verify:** `bash pace/pull.sh && uv run python scripts/report.py --validation runs/georc/scores`,
which writes `results/validation.csv` and prints each G1 check as PASS or FAIL. (Gate checker and
report implemented and unit-tested; awaiting the PACE jobs.)
- 2026-11-10, Fig. 14 rubric (`results/validation_paper_s0.csv`): **G1 FAILS.** F1 is 5-10 points
  under Table 2 for every model, mostly from precision. Judge seed noise is 0.02 F1. Country accuracy
  is 6-8 points low; hedged guesses ("Kazakhstan or Kyrgyzstan") look like the cause. Fallback jobs
  submitted: greedy 15427946, released-code rubric 15427947.
- 2026-11-10 (approved): second country measure, `country_acc_match`. The judge sees the true ISO
  code and says whether the final guess, alternatives included, names it. Both measures are reported;
  whichever reproduces Table 2 is locked in before any generation. Rerun with cached F1: job 15428120.
**Depends on:** T2, T3, T4.
**Files:** `scripts/score_chains.py`, `src/georc/vllm_backend.py`, `pace/score.sbatch`,
`results/validation_paper_s0.csv` (one CSV per judge setting).
**Size:** M

### Checkpoint 1: Scorer validated
- 2026-11-10: the user approved submitting the pilot before this review, to save a queue wait
  (job 15430745: G0, G0R, A1 on 100 images). Full runs still wait for this checkpoint.
- Data kept: `pace/pull.sh` copies chains, scores, logs and the judge cache to `runs/georc/`, so
  nothing has to be regenerated or rejudged after PACE scratch is purged.
- [ ] G1 met, or an agreed fallback recorded in `SPEC.md`.
- [ ] Human reviews `results/validation_*.csv` before any generation job.

---

## Phase 2: Baselines

### T6: Generation pipeline, 100-image pilot
**Description:** `src/georc/generate.py` (vLLM):
- **Model registry:** Gemma at temperature 1.0 with max 512 new tokens; Qwen (deferred) at
  temperature 0.7 with max 32768.
- **Prompts:** G0 uses Fig. 21 verbatim (`gen_paper.txt`). It has one more sentence than the
  released code's prompt ("Output each point on a separate line ... without annotations and
  titles"). G0R uses the released-code prompt (`gen_repo.txt`): all 500 released Gemma chains
  contain bold text or headings, so they were probably generated with it. Whichever of G0 and G0R
  is closer to Table 2 becomes the baseline for G2 and the fixes. We use the chat format; the
  released code fed raw text with no chat template (a documented fidelity risk).
- **Run behavior:** per-chain output files that are skipped if they exist (resume after
  preemption); a per-chain log of input and output tokens, whether it hit max tokens, and GPU time.

The pilot is one PACE job, `pace/pilot.sbatch`: G0, G0R and A1 (pan-and-scan) on challenges 1-20
(100 images), seed 0, then scored, to avoid extra queue waits. It runs only after checkpoint 1,
unless the user says otherwise.
**Acceptance:**
- [ ] 100 chains per condition (G0, G0R, A1). A resubmitted job regenerates nothing that already exists.
- [ ] Pilot F1 of the chosen baseline (G0 or G0R) is within +/- 4 of Table 2 (25.23); the looser
  bound reflects n=100.
- [ ] 10 pilot chains are read side by side with released chains for the same images, and look like the same kind of output.
- 2026-11-10: code done and unit-tested (`conditions.py`, `generate.py`, `generate_chains.py` with
  `--pilot`/`--dry-run`, `pace/generate.sbatch`, `score_chains.py --pilot`). Atomic chain writes make
  preempted jobs safe to resubmit. The pilot job waits for checkpoint 1 (G1 review).
**Verify:** `uv run pytest -q tests/test_generate.py`; `ssh pace 'ls ~/scratch/georc/out/chains/G0/seed0 | wc -l'`
gives 100 (likewise for G0R and A1).
**Depends on:** T5.
**Files:** `src/georc/conditions.py`, `src/georc/generate.py`, `src/georc/templates/gen_paper.txt`,
`src/georc/templates/gen_repo.txt`, `scripts/generate_chains.py`, `pace/generate.sbatch`,
`pace/pilot.sbatch`, `tests/test_generate.py`. **Size:** M

### T7: Full Gemma baseline (gate G2)
**Description:** 500 images x seed 0 for the baseline chosen at the pilot (G0 or G0R), scored.
Results go to `results/baseline.csv`. Q0 is deferred.
**Acceptance (G2):**
- [ ] F1 (seed 0) is within +/- 2.0 of Table 2 (Gemma 25.23), and country accuracy within +/- 4 points (64.0).
- [ ] Truncation rate is reported.
**Verify:** `uv run python scripts/report.py --conditions G0` (or `G0R`).
**Depends on:** T6. **Files:** `src/georc/conditions.py`, `results/baseline.csv`. **Size:** S

### Checkpoint 2: Baselines reproduced
- [ ] G2 met (or the gap explained and accepted).
- [ ] Human approves the Fix B prompt text (B0, B1) before T10's pilot.

---

## Phase 3: Fix A (Gemma resolution)

### T8: Tile mode, pilot
**Description:**
- **A1:** `mm_processor_kwargs={"do_pan_and_scan": True}`. Its pilot now runs in T6's combined
  pilot job (`pace/pilot.sbatch`).
- **A2:** `src/georc/tiles.py`. The full image plus a 2x4 grid of tiles, sent as a 9-image prompt
  with one fixed neutral sentence saying the first image is the full view and the rest are tiles in
  reading order.
- **Pilot:** the 100 images, seed 0, A2.
**Acceptance:**
- [ ] The tiles exactly cover the image, with no gaps or overlaps beyond rounding, and arrive in the stated order (unit test).
- [ ] Logged image-token counts are about 256, 768 and 2,304 for G0, A1 and A2.
- [ ] Pilot chains exist and are scored.
**Verify:** `uv run pytest -q tests/test_tiles.py`; pilot scores printed.
**Depends on:** T7. **Files:** `src/georc/tiles.py`, `src/georc/generate.py`, `src/georc/conditions.py`,
`tests/test_tiles.py`. **Size:** M

### T9: Full Fix A runs
**Description:** A1 and A2 x 500 images x seed 0, scored.
**Acceptance:**
- [ ] 1,000 chains, all scored, with cost logs.
**Verify:** `uv run python scripts/report.py --conditions G0 A1 A2`.
**Depends on:** T8. **Files:** none new (runs only). **Size:** S

---

## Phase 4: Fix B (prompts); can run alongside Phase 3

### T10: B0 and B1 prompts, pilot, freeze
**Description:**
- **B0** (format-only): the baseline prompt chosen at the pilot (Fig. 21 or released-code) plus
  format rules (one clue per line, no introduction or summary, final line "Conclusion: <country>,
  <city>").
- **B1:** B0 plus the content guidance from paper Sec. 2.1.
- **Pilot:** 100 images, seed 0, Gemma (Qwen deferred).
- **Freeze:** after the pilot, write the prompt hashes to `results/prompt_hashes.json`. Full runs
  refuse to start if a hash differs.
**Acceptance:**
- [ ] Unit test: the B1 text is the B0 text plus the guidance block. The B0 text is the baseline prompt text plus the format block.
- [ ] Pilot chains parse into one clue per line in at least 90% of outputs.
- [ ] Hashes are written.
**Verify:** `uv run pytest -q tests/test_generate.py`.
**Depends on:** T7 and the prompt approval at checkpoint 2.
**Files:** prompt template files under `src/georc/templates/`, `src/georc/conditions.py`,
`tests/test_generate.py`, `results/prompt_hashes.json`. **Size:** S

### T11: Full Fix B runs
**Description:** B0 and B1 x Gemma x 500 images x seed 0, scored. Qwen (Q-B0, Q-B1) is deferred.
**Acceptance:**
- [ ] 1,000 chains, all scored. The prompt-hash check passed at the start of every job.
**Verify:** `uv run python scripts/report.py --conditions G0 G0R G-B0 G-B1`.
**Depends on:** T10. **Files:** none new (runs only). **Size:** S

---

## Phase 5: Analysis

### T12: Statistics and report
**Description:** `src/georc/stats.py`:
- **F1:** per-pair F1 (seed 0); the paired difference vs the model's baseline; a cluster
  bootstrap over images (10,000 resamples) for the 95% CI.
- **Country accuracy:** a paired bootstrap of the per-image change.
- **Diagnostics:** truncation rate, and the input and output tokens and GPU-seconds per condition.

`scripts/report.py` writes:
- `results/summary.csv`;
- `results/REPORT.md`, containing a Table 2-style table, the change in F1 and in country accuracy
  with CIs, cost, the verdict per the win rule, the B1 vs B0 content effect, and three example
  chains per fix.
**Acceptance:**
- [ ] The bootstrap recovers a known planted effect on synthetic data (unit test).
- [ ] The win-rule evaluator is tested on both sides of each threshold.
- [ ] REPORT.md states every verdict, including negative ones.
**Verify:** `uv run pytest -q tests/test_stats.py`; human reads `results/REPORT.md`.
**Depends on:** T9, T11.
**Files:** `src/georc/stats.py`, `scripts/report.py`, `tests/test_stats.py`, `results/REPORT.md`, `results/summary.csv`.
**Size:** M

### Checkpoint 3: Results reviewed with human
- [ ] Verdicts agreed. Nothing on PACE is deleted before this checkpoint.

---

## Phase 6: Wrap-up

### T13: Pull results back, verify, clean up PACE
**Description:** `rsync` all chains, scores and logs to the laptop under `runs/georc/` (gitignored)
and verify the counts. Then `pace/cleanup.sh`:
1. **Dry run:** list everything it would remove, and the `georc-*` jobs it would cancel.
2. **Run, after your OK:**
   - `scancel` the `georc-*` jobs only;
   - `rm -rf ~/scratch/georc`;
   - remove the uv binary and uv caches it installed (absent before T4);
   - delete the HF token.
3. **Verify** against `results/pace_baseline.txt`:
   - no `georc-*` jobs left;
   - `~/scratch/georc` gone;
   - home usage back to its starting level or below;
   - director directories and jobs listed identically before and after.
**Acceptance:**
- [ ] The laptop copy has 3,000 chains and all score files (count check).
- [ ] All three cleanup verifications pass, and the output is saved to `results/pace_cleanup.txt`.
**Verify:** `ssh pace 'squeue -u sagrawal394 -h -o %j | grep -c ^georc- ; ls -d ~/scratch/georc'` gives `0` and "No such file".
**Depends on:** checkpoint 3. **Files:** `pace/pull.sh`, `pace/cleanup.sh`, `results/pace_cleanup.txt`. **Size:** S

### Checkpoint 4: Complete
- [ ] All acceptance criteria met; PACE clean; report ready for review.
