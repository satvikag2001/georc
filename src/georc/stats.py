"""Paired cluster bootstrap over images, and the win rule (SPEC.md, Success criteria)."""

import random


def compare(
    base: list[dict], treat: list[dict], field: str, key: tuple[str, ...], scale: float = 1.0
) -> dict[str, list[float]]:
    """Treatment minus baseline per row, grouped by image (the `round` column)."""
    index = {tuple(row[k] for k in key): row for row in treat}
    if len(index) != len(base) or any(tuple(row[k] for k in key) not in index for row in base):
        raise ValueError("baseline and treatment must cover the same rows")
    diffs: dict[str, list[float]] = {}
    for row in base:
        other = index[tuple(row[k] for k in key)]
        diffs.setdefault(row["round"], []).append(scale * (float(other[field]) - float(row[field])))
    return diffs


def paired_bootstrap(
    diffs: dict[str, list[float]], resamples: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean difference and its 95% percentile interval, resampling whole images."""
    clusters = [(sum(values), len(values)) for values in diffs.values()]
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        draw = rng.choices(clusters, k=len(clusters))
        means.append(sum(total for total, _ in draw) / sum(count for _, count in draw))
    means.sort()
    mean = sum(total for total, _ in clusters) / sum(count for _, count in clusters)
    return mean, means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def win(f1_low: float, country_low: float) -> bool:
    """F1 gain's interval excludes 0 and the country change's lower bound is above -2 points."""
    return f1_low > 0 and country_low > -2.0
