"""Country-level accuracy, judged two ways.

The paper measures this "by an LLM judge against the ground truth location country in ISO
format" but does not publish that prompt. Both of ours are checked against Table 2 at gate G1:
  strict (country.txt): the judge extracts one guessed country; the first named wins a hedge.
  match (country_match.txt): the judge sees the true ISO code and says whether the final guess,
    including any alternatives it offers, names that country.
"""

import re
from collections.abc import Sequence

from georc.judge import load_template

_COUNTRY = re.compile(r'"country"\s*:\s*"([A-Za-z]{2})"')
_MATCH = re.compile(r'"match"\s*:\s*(true|false)\b', re.IGNORECASE)


def build_country_match_prompt(chain: str, true_country: str) -> str:
    template = load_template("country_match.txt")
    return template.replace("{country}", true_country.upper()).replace("{chain}", chain)


def parse_match(reply: str) -> bool | None:
    """True or False from {"match": ...}; None for anything else."""
    found = _MATCH.search(reply)
    return None if found is None else found.group(1).lower() == "true"


def build_country_prompt(chain: str) -> str:
    return load_template("country.txt").replace("{chain}", chain)


def parse_country(reply: str) -> str | None:
    """Lower-case ISO alpha-2 code, or None for "XX" or anything unparseable."""
    match = _COUNTRY.search(reply)
    if match is None or match.group(1).upper() == "XX":
        return None
    return match.group(1).lower()


def country_accuracy(predicted: Sequence[str | None], truth: Sequence[str]) -> float:
    """Percent of rounds whose guessed country matches; a missing guess counts as wrong."""
    hits = [p == t for p, t in zip(predicted, truth, strict=True)]
    return 100.0 * sum(hits) / len(hits)
