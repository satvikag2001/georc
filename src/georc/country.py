"""Country-level accuracy: a judge reads the final guess from a chain as an ISO code.

The paper measures this "by an LLM judge against the ground truth location country in ISO
format" but does not publish that prompt; ours is in templates/country.txt and is checked
against the paper's Table 2 accuracies at gate G1.
"""

import re
from collections.abc import Sequence

from georc.judge import load_template

_COUNTRY = re.compile(r'"country"\s*:\s*"([A-Za-z]{2})"')


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
