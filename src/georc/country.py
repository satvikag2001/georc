"""Country accuracy, judged two ways; the paper does not publish its country prompt."""

import re
from collections.abc import Sequence

from georc.judge import load_template

_COUNTRY = re.compile(r'"country"\s*:\s*"([A-Za-z]{2})"')
_MATCH = re.compile(r'"match"\s*:\s*(true|false)\b', re.IGNORECASE)


def build_country_match_prompt(chain: str, true_country: str) -> str:
    """Match measure: the judge sees the true ISO code and says whether the final guess,
    alternatives included, names it."""
    template = load_template("country_match.txt")
    return template.replace("{country}", true_country.upper()).replace("{chain}", chain)


def parse_match(reply: str) -> bool | None:
    found = _MATCH.search(reply)
    return None if found is None else found.group(1).lower() == "true"


def build_country_prompt(chain: str) -> str:
    """Strict measure: the judge extracts one guessed country (the first named, if hedged)."""
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
