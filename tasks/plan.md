# Implementation Plan: GeoRC Fix A (resolution) and Fix B (expert prompt)

Status: approved 2026-11-08; updated 2026-11-10 (Gemma only, one seed, G0R, combined pilot).

## Overview

The goal is to show a measurable improvement on GeoRC (Talreja et al., ACL 2026, Hays lab) for an
open model:

- **Fix A:** give Gemma-3-12B more image resolution. The current pipeline squashes the images (about 2800-3000 x 1500-1600 pixels)
  to 896x896. We test Gemma's built-in pan-and-scan, and a dense tile grid.
- **Fix B:** use an expert-style prompt on Gemma-3-12B. It has a format-only
  control, so formatting gains are not mistaken for better reasoning.

We first rebuild the paper's scorer and check it against the paper's own numbers. The models and the
judge run on PACE. PACE is cleaned up when the work is done. Fix C (self-check) is out of scope for now.

**Scope (2026-11-10):** Gemma-3-12B only, seed 0. The Qwen2.5-VL conditions (Q0, Q-B0, Q-B1) are
deferred and run only if the Gemma results show a difference.

## Facts this plan rests on (checked 2026-11-06 to 11-08)

- **Data.** HF `mohit-talreja/GeoRC` (Apache-2.0, ungated, 5.2 GB): 100 challenges x 5 rounds = 500
  images, plus 800 (round, expert) pairs.
  - Ground-truth country is `streakLocationCode` in `<challenge>_rounds_metadata.json`.
  - Released chains: median about 15 lines and about 520 tokens. One full scoring pass is about
    15,400 judge calls and about 14M prompt tokens.
- **Released scorer is broken.** `score.py` fails in four places: a missing `src.vllm` import, the
  factory receiving the wrong keyword arguments, an f-string format-spec crash in the judge prompt,
  and a `recall` key that never matches. We reimplement it from the paper's Algorithm 1 and Fig. 14.
- **Paper prompts differ from the released code (found in T3).**
  - Judge: Fig. 14's rubric is stricter than the repo's. It scores 0 unless both chains use the same
    evidence type, and 100 only for a direct paraphrase. We use Fig. 14. G1 fallbacks, in order:
    greedy judge decoding, then the repo rubric (extracted verbatim).
  - Generation: Fig. 21 adds "Output each point on a separate line ... without annotations and
    titles". Yet all 500 released Gemma chains contain bold text or headings, so they were probably
    generated with the repo prompt (`gen_repo.txt`). G0 uses Fig. 21 and G0R the repo prompt; the
    pilot picks whichever is closer to Table 2 as the baseline for G2 and the fixes.
  - Input format: the released code fed Gemma raw text through a pipeline with no chat template.
    We use the chat format. This is a second fidelity risk, checked at the pilot and G2.
- **Paper settings (Appendix A.5.1).**
  - Gemma: temperature 1.0, max 512 new tokens.
  - Qwen2.5-VL: temperature 0.7, max new tokens raised to 32768.
  - Judge: Qwen3-4B-Instruct-2507. Its decoding is not stated; the released code uses the model's
    default sampling.
- **Targets from the paper's Table 2 (v2).**
  - Gemma: F1 25.23 (95% CI 24.05 to 26.42), country 64.0%.
  - Qwen2.5-VL: F1 29.09 (CI 28.03 to 30.15), country 69.4%.
  - Llama: F1 24.00, country 58.4%. Qwen3-VL: F1 22.41, country 64.4%. Hallucinated baseline: F1 18.13.
- **PACE.** SSH via `Host pace` works from this laptop. Account `gts-awu36`, QoS `embers`.
  - Compute nodes have no internet, so weights and data are staged from the login node.
  - Home is at 14.3 of 20 GB, so all caches go on scratch.
  - Scratch already holds 1.6 TB of director-project data, and 41 director jobs are queued.
    Neither may be touched.
  - GPU partitions: a100 (13 nodes), h100, h200, l40s, all busy. CUDA modules go up to 13.0.1.

## Architecture decisions

- **All GPU work runs on PACE.** That covers both models and the judge, in one environment. The
  laptop is used only for code, CPU unit tests and analysis.
- **vLLM offline batch inference for everything.** Plain transformers at batch size 1 would take
  about 25 h for generation; vLLM takes about 1 h. vLLM's prefix caching also makes the judge cheap,
  because its long rubric is a shared prefix.
- **Faithful first, improvements second.** We use the paper's prompts verbatim (Fig. 21 for
  generation, Fig. 14 for the judge), its line-splitting rules and its decoding settings.
  - The only documented deviation: the judge's score is parsed from the first number in its JSON
    reply, because the released code's key lookup sets recall to 0.
  - Documented alternatives: G0R (repo generation prompt) and the G1 judge fallbacks.
- **Paired design.** Every condition uses the same 500 images and one seed (0; decided 2026-11-10 to
  save time and compute). Each chain is one independent draw, so the bootstrap still covers
  generation randomness, with wider intervals than 3 seeds would give. Differences are
  computed per (round, expert) pair, then bootstrapped by image, so the 3 experts on a shared image
  are not counted as independent.
- **Equal output budgets within a model.** Every Gemma condition keeps 512 max new tokens. The rate
  of truncated chains is reported, because a median chain is about 520 tokens.
- **Prompts are frozen before full runs.** After each pilot, prompt text is hashed. A full run
  refuses to start if the hash changed, so no prompt is tuned on the evaluation set after seeing
  full results.
- **One PACE footprint.** Everything lives under `~/scratch/georc/` (code, venv, `HF_HOME`, uv
  cache, data, outputs, logs). Every job is named `georc-*`.
  - Cleanup removes exactly that directory and those jobs.
  - It also removes uv, which is not installed on PACE today and is only installed for this work.

## Conditions (3,000 chains: 6 Gemma conditions x 500 images, seed 0)

| Id | Model | Change vs paper baseline | Image tokens | Seeds |
|---|---|---|---|---|
| G0 | Gemma-3-12B | none (896 px squash; Fig. 21 prompt) | ~256 | 1 |
| G0R | Gemma-3-12B | released-code prompt (`gen_repo.txt`) | ~256 | 1 |
| A1 | Gemma-3-12B | pan-and-scan (`do_pan_and_scan`) | ~768 | 1 |
| A2 | Gemma-3-12B | full view + 2x4 tiles, plus one neutral sentence saying so | ~2,304 | 1 |
| G-B0 | Gemma-3-12B | format-only prompt (control) | ~256 | 1 |
| G-B1 | Gemma-3-12B | expert prompt | ~256 | 1 |
| Q0 | Qwen2.5-VL-7B | none | ~5,800 (native) | deferred |
| Q-B0 | Qwen2.5-VL-7B | format-only prompt (control) | ~5,800 | deferred |
| Q-B1 | Qwen2.5-VL-7B | expert prompt | ~5,800 | deferred |

- **Baseline choice:** the pilot runs G0 and G0R on the same 100 images; whichever is closer to
  Table 2 becomes the Gemma baseline for G2 and for the fixes.
- **Deferred:** Q0, Q-B0 and Q-B1 (1,500 more chains) run only if the Gemma results show a difference.

- The **format-only prompt (B0)** is the paper prompt plus: one clue per line, no introduction or
  summary, and a final line "Conclusion: <country>, <city>".
- The **expert prompt (B1)** is B0 plus content guidance from paper Sec. 2.1:
  - only discriminative clues, each naming where that clue occurs;
  - coarse to fine;
  - weigh alternatives before committing, which counters misattribution;
  - no claimed tools, which counters false tool use;
  - nothing true-everywhere, which counters axiomatic irrelevance.
- **How Fix B is read:** the B1 vs baseline difference is the gain, and B1 vs B0 is the part of the
  gain that comes from content rather than format.

## Dependency graph

```
T1 spec ──────────────────────────────────────────────┐
T2 data + parsing ──┐                                 │
T3 scorer (fake judge) ──┤                            │
T4 PACE env + assets ────┴─> T5 validate scorer (G1) ─┴─> T6 generation + pilot ─> T7 full baseline (G2)
                                                                                   │
                                     ┌─────────────────────────────────────────────┤
                                     v                                             v
                         T8 Fix A tiles + pilot ─> T9 Fix A full    T10 Fix B prompts + pilot ─> T11 Fix B full
                                     └──────────────────────┬──────────────────────┘
                                                            v
                                                T12 stats + report ─> T13 pull back + PACE cleanup
```

T2, T3 and T4 can proceed in parallel. T6's pilot is one PACE job (`pace/pilot.sbatch`) covering
G0, G0R and A1, so A1's pilot no longer waits for T8. After checkpoint 2, Fix A (T8-T9) and Fix B
(T10-T11) are independent.

## Task list (details and acceptance criteria in `tasks/todo.md`)

### Phase 0: Spec
- [x] T1: GeoRC `SPEC.md`; park the 3DGPE spec as `specs/3dgpe.md`

### Phase 1: A trustworthy scorer (riskiest, so first)
- [x] T2: Dataset fetch, challenge/round/expert parsing, point extraction (local, tested)
- [x] T3: One-to-all scorer and country check with a pluggable judge (local, fake judge)
- [x] T4: PACE environment, asset staging, preflight job
- [ ] T5: Re-score released chains on PACE and match Table 2 (**gate G1**)

### Checkpoint 1: Scorer validated
- [ ] G1 met; human reviews `results/validation_*.csv` before any generation

### Phase 2: Baselines
- [ ] T6: vLLM generation pipeline; one pilot job (G0, G0R, A1 on 100 images, seed 0, scored)
- [ ] T7: Full Gemma baseline (G0 or G0R, as the pilot picks), seed 0 (**gate G2**); Q0 deferred

### Checkpoint 2: Baselines reproduced
- [ ] G2 met; human reviews; Fix B prompt text approved before T10's pilot

### Phase 3: Fix A (Gemma resolution)
- [ ] T8: Tile mode (A2), 100-image pilot (A1's pilot moved into T6)
- [ ] T9: Full A1 and A2 runs, scored

### Phase 4: Fix B (prompts; can run alongside Phase 3)
- [ ] T10: B0 and B1 prompts, Gemma pilot, then freeze
- [ ] T11: Full B0 and B1 runs for Gemma, scored (Qwen deferred)

### Phase 5: Analysis
- [ ] T12: Paired bootstrap statistics, `results/REPORT.md`

### Checkpoint 3: Results reviewed with human

### Phase 6: Wrap-up
- [ ] T13: Pull results back, verify, clean up PACE

## Success criteria

- **G1 (scorer):**
  - On the released chains, each open model's F1 is within its paper 95% CI +/- 1.0.
  - The ranking Qwen2.5-VL > Gemma > Llama > Qwen3-VL holds.
  - Country accuracy is within +/- 3 points of Table 2.
  - The hallucinated baseline is within +/- 2 of 18.13.
  - Two country measures: strict (`country_acc`) and match (`country_acc_match`, where the judge
    sees the true ISO code). Fallbacks: greedy judge decoding, then the repo rubric. The combination
    that reproduces Table 2 is locked in before any generation.
- **G2 (baseline):** F1 (seed 0) of the chosen Gemma baseline (G0 or G0R) is within +/- 2.0 of
  Table 2 (25.23), and country accuracy within +/- 4 points (64.0).
- **Win (confirmed):** the condition's F1 gain over its baseline has a 95%
  CI that excludes 0, **and** the country-accuracy change has a 95% CI lower bound above -2 points.
  - Reported for A1, A2 and G-B1 (Q-B1 deferred).
  - B1 vs B0 is reported alongside, to separate content from format.
  - Cost (input and output tokens, GPU time) is reported next to every number.
- **Done:**
  - `results/REPORT.md` and `results/summary.csv` are committed.
  - All 3,000 chains and their scores are copied to the laptop.
  - PACE holds nothing from this project.

## Compute budget (estimate; Gemma only, seed 0)

| Item | Amount | A100-hours |
|---|---|---|
| Generation | 3,000 chains, at most ~1.5M output tokens (512 per chain) | ~0.3-0.5 |
| Scoring | ~10 passes x ~15k judge calls (6 conditions, plus validation reruns) | 1-2 |
| Pilots and preflight | | ~0.5 |
| **Total** | | **~2-3 A100-hours over ~12 jobs** |

- The deferred Qwen conditions are not included.

- Calendar time is dominated by the `embers` queue: about 1-2 weeks overall.
- Disk on PACE: about 55 GB (weights ~50 GB, data 5.2 GB, outputs under 1 GB). All of it is removed at the end.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Rebuilt scorer can't reproduce Table 2 (judge decoding not stated) | High | Happening: G1 fails with the Fig. 14 rubric (F1 5-10 points low, country 6-8 points low). G1 runs before any generation. Fallbacks in order: greedy judge decoding, then the repo rubric; both country measures (strict, match) are reported. The combination that reproduces Table 2 is locked in before any generation. If none does, stop and ask the GeoRC authors. A preserved ranking still allows relative comparisons, with that caveat stated. |
| vLLM preprocessing differs from the paper's transformers pipeline | Med | G2 gate, plus side-by-side reading of 10 chains. Processor defaults stay unchanged for G0 and G0R. |
| Released Gemma chains came from a different prompt or input format (all 500 have bold text or headings, which Fig. 21 forbids; the released code used raw text with no chat template) | Med | Pilot runs G0 (Fig. 21) and G0R (repo prompt) on the same images; the one closer to Table 2 becomes the baseline. The chat-format difference is documented and checked at G2. |
| Qwen's 32,768-token output limit could exceed the judge's context (deferred) | Med | Add a guard (flag or cap over-long chains before scoring) before any Qwen run. |
| vLLM lacks Gemma pan-and-scan | Med | T4 preflight tests it. Fallback: compute the crops ourselves through the A2 multi-image path. |
| `embers` preemption and busy queues; the 41 director jobs share the fair-share | Med | Each chain is written to disk as soon as it's done, and a resubmitted job skips existing ones. Request `gpu:1` across a100,h100,h200,l40s. Run a pilot before every full run. |
| Gemma truncation at 512 tokens (median chain ~520 tokens) | Med | Same budget in every Gemma condition. Truncation rate reported per condition, since B1 may raise country accuracy partly by finishing chains. |
| Fix B looks like gaming the scorer | Med | B0 format-only control. B1 vs B0 reported. Prompts frozen by hash before full runs. |
| Tiles in A2 add a sentence to the prompt (confound) | Low | One neutral sentence, recorded. A1 has no prompt change. |
| Home quota (14.3/20 GB) | Med | `HF_HOME`, `UV_CACHE_DIR`, `TMPDIR` and the pip cache all on scratch. Home usage checked before and after. |
| No internet on compute nodes | High if missed | Assets staged on the login node. `HF_HUB_OFFLINE=1` in jobs. Preflight proves offline loading. |
| Cleanup removes the wrong thing | High | `pace/cleanup.sh` only deletes `~/scratch/georc` and the uv binary and caches it installed. It runs a dry-run listing first and needs your confirmation. Director directories and jobs are listed before and after to prove they are untouched. |
| Hugging Face rate limits (seen: HTTP 429) | Low | Use an HF token for all downloads. |

## Resolved decisions

1. **Win rule:** accepted as stated above.
2. **Cost:** Fix A's extra inference (about 3x and 9x the image tokens) is accepted if reported.
3. **PACE jobs:** submitted from this laptop on `embers` under `gts-awu36`, job names `georc-*`.
4. **Hugging Face:** Gemma license accepted; the token is stored only on PACE scratch
   (`~/scratch/georc/.hf_token`, mode 600) and deleted at cleanup.
5. **Commits:** commit after each task and push to `origin` (`satvikag2001/georc`, public).
   Messages under 30 words, no AI attribution.
