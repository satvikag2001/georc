"""Split reasoning chains into the points the GeoRC judge scores."""

import re

_BULLET = re.compile(r"^(\d+\.|\*|-)\s*")


def extract_points(chain: str) -> list[str]:
    """Split a chain into scorable points exactly as the paper's one-to-all scorer does.

    One point per non-empty line, with a leading "1.", "*" or "-" removed; fragments of five
    characters or fewer are dropped (GeoRC Algorithm 1; released score.py `_extract_points`).
    """
    lines = (line.strip() for line in chain.splitlines())
    points = (_BULLET.sub("", line).strip() for line in lines if line)
    return [p for p in points if len(p) > 5]
