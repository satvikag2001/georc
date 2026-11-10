# georc

Tests two fixes for open vision-language models on GeoRC, a benchmark for geolocation reasoning
chains (ACL 2026, arXiv 2601.21278). A model looks at a street-level image, writes a reasoning
chain, and is scored against human expert chains by an LLM judge (F1 over matched points, plus
country accuracy). The goal is a gain that clearly exceeds noise and that the GeoRC authors can
verify from this repository.

## The fixes (Gemma-3-12B)

- **Fix A, resolution.** Gemma squashes each image (about 2800-3000 x 1500-1600 pixels) to
  896x896. A1 uses Gemma's built-in pan-and-scan; A2 sends the full view plus a 2x4 tile grid.
- **Fix B, expert-style prompt.** B1 asks for the chain structure the paper defines (Sec. 2.1).
  B0 is a format-only control, so formatting gains are not mistaken for better reasoning.

Baselines: G0 uses the paper's printed generation prompt (Fig. 21); G0R uses the prompt from the
authors' released code. Whichever is closer to the paper's Table 2 becomes the baseline. Qwen2.5-VL
conditions are deferred until the Gemma results show a difference.

## Status

- Scorer validation against the paper's Table 2 is in progress and has not yet reproduced it: with
  the paper's judge rubric (Fig. 14), F1 is 5-10 points low. Fallback judge settings are being tried.
- No fix results yet. Generation starts only after the scorer is validated.

## Running the tests

```bash
uv sync && uv run pytest -q
```

GPU work (generation and judging with vLLM) runs on a Slurm cluster through the scripts in
`pace/`; see `SPEC.md` for the commands.

## Documents

- `SPEC.md`: objective, gates, win rule, boundaries (the source of truth).
- `tasks/plan.md`: design decisions, conditions, compute budget, risks.
- `tasks/todo.md`: task list with acceptance criteria and status.
- `THIRD_PARTY_NOTICES.md`: material copied or transcribed from the GeoRC paper and code.

## License

No license has been chosen yet. Third-party material keeps its own license (see
`THIRD_PARTY_NOTICES.md`).

## Citation

Talreja, Diao, James, Casapu, Santanam, Mendes, Ritter, Xu and Hays. *GeoRC: A Benchmark for
Geolocation Reasoning Chains.* ACL 2026. arXiv:2601.21278.
