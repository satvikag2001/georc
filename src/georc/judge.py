"""Judge prompts, reply parsing and judge backends for GeoRC one-to-all scoring."""

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Protocol

# "paper": the rubric printed in GeoRC Fig. 14 (default; the paper wins over the code).
# "repo": the rubric in the released score.py, extracted verbatim; fallback for gate G1.
PROMPT_VARIANTS = {"paper": "judge_paper.txt", "repo": "judge_repo.txt"}

_JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


@cache
def load_template(name: str) -> str:
    return (files("georc") / "templates" / name).read_text().removesuffix("\n")


def build_prompt(first_chain: str, second_point: str, variant: str = "paper") -> str:
    """Fill a judge rubric. Plain replacement, so braces in chain text are left alone."""
    template = load_template(PROMPT_VARIANTS[variant])
    return template.replace("{first_chain}", first_chain).replace("{second_point}", second_point)


def parse_score(reply: str) -> float | None:
    """First number inside the first {...} of a judge reply, clipped to 0-100.

    Deviation from the released code, which read the key "precision" or "recall" and so scored
    every recall call 0 (both rubrics show only a "precision" example). None means unparseable.
    """
    obj = _JSON_OBJECT.search(reply)
    if obj is None:
        return None
    number = _NUMBER.search(obj.group(0))
    if number is None:
        return None
    return min(100.0, max(0.0, float(number.group(0))))


class Judge(Protocol):
    def complete(self, prompts: Sequence[str]) -> list[str]:
        """Raw replies, one per prompt, in order."""
        ...


class FunctionJudge:
    """A judge backed by a plain function; for tests and dry runs."""

    def __init__(self, reply: Callable[[str], str]) -> None:
        self._reply = reply

    def complete(self, prompts: Sequence[str]) -> list[str]:
        return [self._reply(p) for p in prompts]


class CachedJudge:
    """Wraps a judge with an append-only JSONL cache keyed by prompt hash.

    Uncached prompts are sent in chunks and each finished chunk is written before the next one
    starts, so a preempted job loses at most one chunk. Use one cache file per judge model,
    rubric variant and sampling seed.
    """

    def __init__(self, inner: Judge, path: Path, chunk_size: int = 2000) -> None:
        self._inner = inner
        self._path = Path(path)
        self._chunk_size = chunk_size
        self._replies: dict[str, str] = {}
        if self._path.exists():
            for line in self._path.read_text().splitlines():
                record = json.loads(line)
                self._replies[record["key"]] = record["reply"]

    @staticmethod
    def _key(prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()

    def complete(self, prompts: Sequence[str]) -> list[str]:
        keys = [self._key(p) for p in prompts]
        todo = list({k: p for k, p in zip(keys, prompts, strict=True) if k not in self._replies})
        prompt_by_key = dict(zip(keys, prompts, strict=True))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(todo), self._chunk_size):
            chunk = todo[start : start + self._chunk_size]
            replies = self._inner.complete([prompt_by_key[k] for k in chunk])
            with self._path.open("a") as f:
                for key, reply in zip(chunk, replies, strict=True):
                    f.write(json.dumps({"key": key, "reply": reply}) + "\n")
                    self._replies[key] = reply
        return [self._replies[k] for k in keys]
