# Spec: Improving open-VLM geolocation reasoning on GeoRC

Status: approved 2026-11-08; updated 2026-11-10 (Gemma-3-12B only, one seed, baseline G0R added).
Plan: `tasks/plan.md`, tasks: `tasks/todo.md`. Parked earlier study: `specs/3dgpe.md`.

## Objective

**Goal.** Demonstrate a measurable improvement on GeoRC, a current benchmark paper. Even a marginal
improvement counts, if it clearly exceeds noise and the paper's authors can verify it.

**Paper.** *GeoRC: A Benchmark for Geolocation Reasoning Chains* (Talreja, Diao, James, Casapu,
Santanam, Mendes, Ritter, Xu, Hays; ACL 2026; arXiv 2601.21278). Data: HF `mohit-talreja/GeoRC`
(Apache-2.0). Code: github.com/talrejamohit03/GeoRC.

**Audience.** The GeoRC authors, who should be able to verify every number from this repository.

**What we test.** Two fixes, each aimed at a failure type the paper names:
- **Fix A, resolution (Gemma-3-12B).** The paper hypothesizes that open VLMs fail at "extracting
  fine-grained visual attributes from high resolution images" (abstract, Figs. 8 and 25). Gemma
  squashes each image (about 2800-3000 x 1500-1600 pixels; sizes vary) to 896x896. We test the built-in pan-and-scan (A1) and a full view
  plus a 2x4 tile grid (A2).
- **Fix B, expert-style prompt (Gemma-3-12B).** The paper notes that every model
  got one fixed prompt and "could perform more competitively ... with specific prompt tuning"
  (Sec. 7). It also names the failures axiomatic irrelevance, misattribution and false tool use.
  B1 asks for the chain structure the paper defines in Sec. 2.1. B0, a format-only control,
  separates formatting gains from reasoning gains.

**Scope (2026-11-10).** Gemma-3-12B only, seed 0: six conditions (G0, G0R, A1, A2, G-B0, G-B1),
3,000 chains at 500 images each. The Qwen2.5-VL conditions (Q0, Q-B0, Q-B1) are deferred and run
only if the Gemma results show a difference.

**Which baseline.** G0 uses the paper's printed prompt (Fig. 21, "without annotations and titles").
G0R uses the prompt from the authors' released code (`templates/gen_repo.txt`). All 500 released
Gemma chains contain bold text or headings, which Fig. 21 forbids, so they were probably generated
with the released-code prompt. The pilot runs both on the same 100 images; the one closer to Table 2
becomes the baseline for G2 and for the fixes. A second fidelity risk: the released code fed Gemma
raw text through a pipeline with no chat template, while we use the chat format.

**Out of scope for now.** Fix C (self-check), combining A and B, closed models, any training.

## Tech stack

- Python 3.12, managed with `uv`.
- Local: numpy, pillow, huggingface_hub, pytest, ruff.
- PACE: the above plus vLLM 0.32.0 (torch 2.13, CUDA 13.0).
- Models:
  - `google/gemma-3-12b-it` (gated; bf16, 24.4 GB)
  - `Qwen/Qwen2.5-VL-7B-Instruct` (16.6 GB; staged, but its conditions are deferred)
  - judge `Qwen/Qwen3-4B-Instruct-2507` (8.0 GB)
- Compute: PACE Phoenix, account `gts-awu36`, QoS `embers` (free), SSH host `pace`.

## Commands

```bash
# Local
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
uv run python scripts/fetch_dataset.py --out data/georc --text-only     # chains + metadata for local dev

# PACE (all paths under ~/scratch/georc/{code,venv,hf,uv,data,out,logs})
bash pace/sync.sh                                           # push code only
ssh pace 'bash ~/scratch/georc/code/pace/setup_env.sh'      # login node: uv, Python 3.12, vLLM 0.32.0
ssh pace 'bash ~/scratch/georc/code/pace/fetch_assets.sh'   # login node: dataset + 3 models (needs HF token)
bash pace/submit.sh preflight.sbatch
bash pace/submit.sh score.sbatch    --released qwen gemma3_12b llama3_11b qwen3 hallucinated
bash pace/submit.sh pilot.sbatch                            # G0, G0R, A1 on 100 pilot images, seed 0, scored
bash pace/submit.sh generate.sbatch --condition G0 --seeds 0
bash pace/pull.sh                                           # copy outputs to runs/georc/
uv run python scripts/report.py --conditions G0 G0R A1 A2 G-B0 G-B1
bash pace/cleanup.sh --dry-run && bash pace/cleanup.sh      # only after checkpoint 3
```

## Project structure

```
src/georc/        data.py points.py judge.py score.py country.py evaluate.py gates.py models.py
                  conditions.py generate.py vllm_backend.py
                  templates/ (generation, judge and country prompts as text files)
                  planned: tiles.py stats.py
scripts/          fetch_dataset.py preflight.py score_chains.py generate_chains.py report.py
pace/             common.sh setup_env.sh fetch_assets.sh sync.sh submit.sh pull.sh
                  preflight.sbatch score.sbatch generate.sbatch pilot.sbatch; planned: cleanup.sh
tests/            test_*.py (no fixtures directory)
results/          committed: validation_*.csv, baseline.csv, summary.csv, REPORT.md, prompt_hashes.json;
                  local only (gitignored, holds PACE account details): pace_baseline.txt, pace_cleanup.txt
specs/            parked specs
tasks/            plan.md, todo.md
data/, runs/      gitignored: dataset copy, generated chains and judge outputs
```

## Code style

Python 3.12, type hints on public functions, `ruff format` (line length 100), small pure functions.
Anything that reproduces the paper says so in its docstring, and cites the figure or section.
Deviations from the released code are named in a comment.

```python
_BULLET = re.compile(r"^(\d+\.|\*|-)\s*")


def extract_points(chain: str) -> list[str]:
    """Split a chain into scorable points exactly as the paper's one-to-all scorer does.

    One point per non-empty line, with a leading "1.", "*" or "-" removed; fragments of five
    characters or fewer are dropped (GeoRC Algorithm 1; released score.py `_extract_points`).
    """
    lines = (line.strip() for line in chain.splitlines())
    points = (_BULLET.sub("", line).strip() for line in lines if line)
    return [p for p in points if len(p) > 5]
```

## Testing strategy

- **Unit tests (pytest, CPU, under a minute):** parsing, point extraction, scoring arithmetic with a
  deterministic `FakeJudge`, judge-reply parsing edge cases, tile geometry, prompt composition,
  bootstrap and the win-rule evaluator.
- **GPU code is never run locally.** Correctness on PACE is established by the gates:
  - preflight (T4): models load offline and generate;
  - G1 (T5): the scorer reproduces the paper;
  - G2 (T7): the baselines reproduce the paper.
- **Rule:** every public function in `src/georc` has a test. `stats.py` and the win-rule evaluator
  get full branch coverage, because they decide the verdict.

## Success criteria

- **G1, scorer:** on the released chains, each open model's F1 is within its paper 95% CI +/- 1.0.
  The ranking Qwen2.5-VL > Gemma > Llama > Qwen3-VL holds. Country accuracy is within +/- 3 points
  of Table 2. The hallucinated baseline's F1 is within +/- 2 of 18.13.
  - Two country measures are reported: strict (`country_acc`: the judge extracts one guessed
    country) and match (`country_acc_match`: the judge sees the true ISO code and says whether the
    final guess, alternatives included, names it).
  - If G1 fails with the Fig. 14 rubric, try greedy judge decoding, then the released-code rubric.
    The combination that reproduces Table 2 is locked in before any generation.
  - Status 2026-11-10: G1 fails with the Fig. 14 rubric (F1 5-10 points low); fallback jobs queued.
- **G2, baseline:** F1 of the chosen Gemma baseline (G0 or G0R; seed 0, one chain per image, as in
  the paper) is within +/- 2.0 of Table 2 (25.23), and country accuracy within +/- 4 points (64.0).
- **Win, per condition vs its model's baseline:**
  - the F1 gain's 95% CI (cluster bootstrap over images) excludes 0, **and**
  - the country-accuracy change's 95% CI lower bound is above -2 points;
  - cost (tokens, GPU time) is reported next to every number.
- **Pilot images.** Headline results use all 500 images. The same verdicts are also reported
  without the 100 pilot images, because the pilot saw them.
- **Done:**
  - `results/REPORT.md` reports every verdict, including negative ones.
  - All chains and scores are on the laptop.
  - PACE holds nothing from this project (`results/pace_cleanup.txt`).

## Boundaries

**Always**
- Run `ruff` and `pytest -q` before each commit. Commit after each task and push to `origin`
  (`satvikag2001/georc`, public). Messages are under 30 words, with no AI attribution.
- Use the paper's prompts verbatim (Fig. 21 for generation, Fig. 14 for the judge), its parsing
  rules and its decoding settings for every baseline and every validation run. The only exceptions
  are the documented ones: G0R's released-code prompt and the G1 judge fallbacks.
- Fixed seeds. Every chain is written to disk as soon as it's done, and resubmitted jobs skip
  existing outputs.
- Freeze prompt hashes after each pilot. Full runs refuse to start if a hash changed.
- PACE:
  - only `~/scratch/georc/`, with every cache on scratch;
  - job names `georc-*`, account `gts-awu36`, QoS `embers`;
  - `HF_HUB_OFFLINE=1` inside jobs.
- Report every run in the grid.

**Ask first**
- Changing a gate, the win rule, the conditions, the seeds or a frozen prompt.
- Adding conditions (Fix C, A plus B combined).
- Running `pace/cleanup.sh` for real; it shows a dry run first.
- Cancelling any job.
- Contacting the authors.

**Never**
- Use `gts-awu36-paid`, the `inferno` QoS, or any paid compute, credits or API.
- Read, modify or delete director-project files, or touch its jobs.
- Put the HF token in git, in home, or in logs.
- Tune a prompt after seeing full-run results.
- Weaken or skip a failing test to get a run through.

## Open questions

- None blocking. The Gemma license was granted and the HF token is stored only on PACE scratch.
