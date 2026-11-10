"""Run the GeoRC authors' own released scorer on the released chain sets (PACE).

  authors_score.py --authors A --data D --out O [--tags qwen3] [--shard 1/2]
  authors_score.py --data D --out O --status       # progress and summaries; no GPU, no torch

The scoring is their code: EvaluationPipeline and BipartiteLLMScorer from score.py at commit
a7a1679 plus patches/authors_a7a1679.patch (cloned by pace/fetch_authors.sh), with the judge
loaded by their get_processor_and_model("qwen_llm"). This script only lays the dataset out the
way their code expects (georc.authors.build_view), loads the judge once, records every raw judge
reply, and checkpoints. Their scorer makes one generate() call per point, about 77,000 for the
five released sets, so the work is split into units of five pairs. Each unit is written as soon
as it is done and skipped on resubmission. After --budget-minutes no new unit starts and the job
exits 0, so jobs can be chained with submit.sh --after.

Writes O/<run>/units/<tag>/<expert>/<challenge>.json (their per-pair records plus raw replies)
and, once every unit of a tag is done, O/<run>/<tag>/{pairs.json,summary.json}; <run> is
<dtype>_s<seed>, or "fake" with --fake-judge.
"""

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

from georc import authors
from georc.data import RELEASED_TAGS
from georc.models import JUDGE


class Recorder:
    """Keeps each judge reply, grouped by scored pair. Arguments and results pass unchanged."""

    def __init__(self, hf_module, scorer) -> None:
        self.pairs: list[list[dict]] = []
        self._calls: list[dict] = []
        self._reply = ""
        text_to_text = hf_module.text_to_text
        get_score, score = scorer._get_llm_score, scorer.score

        def recorded_text_to_text(*args, **kwargs):
            self._reply = text_to_text(*args, **kwargs)
            return self._reply

        def recorded_get_score(first_chain, second_point, metric_type):
            value = get_score(first_chain, second_point, metric_type)
            self._calls.append({"metric": metric_type, "reply": self._reply, "value": value})
            return value

        def recorded_score(candidate, ground_truth):
            self._calls = []
            result = score(candidate, ground_truth)
            self.pairs.append(self._calls)
            return result

        # _get_llm_score imports text_to_text from the module at call time, so this is seen.
        hf_module.text_to_text = recorded_text_to_text
        scorer._get_llm_score, scorer.score = recorded_get_score, recorded_score


def fake_text_to_text(prompt, model, tokenizer):
    """Stand-in judge for --fake-judge: a fixed pseudo-random score per prompt."""
    return json.dumps({"precision": zlib.crc32(prompt.encode()) % 101})


def load_authors(args) -> tuple:
    """Import their score.py and judge helpers, load the judge, build their bipartite scorer."""
    sys.path.insert(0, str(args.authors))
    os.environ["GEORC_JUDGE_REVISION"] = JUDGE.revision  # patched loader: pinned cached commit
    os.environ["GEORC_SCORE_KEY"] = args.score_key
    import score as their_score
    import src.hugging_face.hf_completion_utils as hf

    for module in (their_score, hf):
        if not Path(module.__file__).resolve().is_relative_to(args.authors.resolve()):
            raise SystemExit(f"{module.__name__} came from {module.__file__}, not {args.authors}")
    if args.fake_judge:
        hf.text_to_text, llm, tokenizer, info = fake_text_to_text, None, None, {"judge": "fake"}
    else:
        import torch
        import transformers

        llm, tokenizer = hf.get_processor_and_model("qwen_llm", args.cache_dir, device_id=0)
        loaded = str(llm.dtype)
        if args.dtype != "native":
            llm = llm.to(getattr(torch, args.dtype))
        info = {
            "judge": f"{JUDGE.repo}@{JUDGE.revision}",
            "dtype_as_loaded": loaded,
            "dtype": str(llm.dtype),
            "generation_config": llm.generation_config.to_dict(),
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        }
    scorer = their_score.ScorerFactory.get_scorer(
        "bipartite", llm_model=llm, llm_tokenizer=tokenizer
    )
    print(info, flush=True)
    return their_score, scorer, Recorder(hf, scorer), info


def score_unit(unit: authors.Unit, args, run_dir: Path, their_score, scorer, recorder) -> None:
    view = authors.build_view(args.data, args.out / "views", unit)
    if not args.fake_judge:
        import transformers

        transformers.set_seed(authors.unit_seed(unit, args.seed))
    recorder.pairs = []
    pipeline = their_score.EvaluationPipeline(scorer, output_path=None)
    start = time.monotonic()
    pipeline.run(str(view), authors.candidate_pattern(unit.tag))
    if len(pipeline.results) != authors.ROUNDS or len(recorder.pairs) != authors.ROUNDS:
        raise RuntimeError(f"{unit}: {len(pipeline.results)} results, expected {authors.ROUNDS}")
    record = {
        "tag": unit.tag,
        "expert": unit.expert,
        "challenge": unit.challenge,
        "score_key": args.score_key,
        "seconds": round(time.monotonic() - start, 1),
        "results": pipeline.results,
        "calls": recorder.pairs,
    }
    authors.write_json(unit.path(run_dir), record)


def write_summaries(args, run_dir: Path) -> None:
    """Merge and summarize every tag whose units are all done (by any shard or job)."""
    for tag in args.tags:
        units = authors.list_units(args.data, [tag])
        done = sum(u.path(run_dir).exists() for u in units)
        if done < len(units):
            print(f"{tag:14s} {done}/{len(units)} units done")
            continue
        rows = authors.load_pairs(run_dir, units)
        keys = sorted({r["score_key"] for r in rows})
        if len(keys) > 1:
            print(f"{tag:14s} not summarized: units mix score keys {keys}; use a fresh --out")
            continue
        pairs = [{k: v for k, v in r.items() if k not in ("calls", "score_key")} for r in rows]
        summary = {"tag": tag, "score_key": keys[0], **authors.summarize(pairs)}
        authors.write_json(run_dir / tag / "pairs.json", pairs)
        authors.write_json(run_dir / tag / "summary.json", summary)
        print(
            f"{tag:14s} P {summary['precision']:.2f} R {summary['recall']:.2f}"
            f" F1 {summary['f1']:.2f} (n={summary['n']})"
        )


def vars_json(args) -> dict:
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authors", type=Path, help="patched clone (pace/fetch_authors.sh)")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tags", nargs="+", choices=RELEASED_TAGS, default=list(authors.TAGS))
    parser.add_argument("--shard", default="1/1", help="i/n: every n-th unit, from the i-th")
    parser.add_argument("--budget-minutes", type=float, default=105)
    parser.add_argument("--limit", type=int, help="score at most this many units (smoke test)")
    parser.add_argument("--cache-dir", help="HF hub cache passed to their loader")
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "native"],
        default="float32",
        help="judge weights; float32 is what their unpinned from_pretrained gave under"
        " transformers 4.x, native keeps what the installed transformers loads",
    )
    parser.add_argument("--score-key", choices=authors.SCORE_KEYS, default="fallback")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fake-judge", action="store_true", help="no model; fixed fake replies")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()

    run_dir = args.out / ("fake" if args.fake_judge else f"{args.dtype}_s{args.seed}")
    units = authors.shard(authors.list_units(args.data, args.tags), args.shard)
    todo = [u for u in units if not u.path(run_dir).exists()]
    print(f"{run_dir}: shard {args.shard} has {len(units)} units, {len(todo)} to do", flush=True)
    if args.limit is not None:
        todo = todo[: args.limit]
    if todo and not args.status:
        if args.authors is None:
            parser.error("--authors is required to score")
        their_score, scorer, recorder, info = load_authors(args)
        job = os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")
        authors.write_json(run_dir / f"run_{job}.json", {**vars_json(args), **info})
        for i, unit in enumerate(todo):
            if time.monotonic() - started > 60 * args.budget_minutes:
                print(f"budget reached; {len(todo) - i} units left: resubmit", flush=True)
                break
            score_unit(unit, args, run_dir, their_score, scorer, recorder)
            print(f"done {unit} ({i + 1}/{len(todo)})", flush=True)
    write_summaries(args, run_dir)


if __name__ == "__main__":
    main()
