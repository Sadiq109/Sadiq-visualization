"""Load the Iris CSV and compute summary statistics."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

FEATURES = ("sepal_length", "sepal_width", "petal_length", "petal_width")
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "iris.csv"


@dataclass(frozen=True)
class Sample:
    features: tuple[float, float, float, float]
    species: str


def load(path: Path | str = DEFAULT_PATH) -> list[Sample]:
    """Read the CSV into samples. Raises ValueError on a bad row."""
    samples: list[Sample] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = set(FEATURES + ("species",)) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing columns: {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            try:
                values = tuple(float(row[name]) for name in FEATURES)
            except ValueError as exc:
                raise ValueError(f"line {line}: {exc}") from None
            samples.append(Sample(values, row["species"].strip()))
    if not samples:
        raise ValueError("no rows in dataset")
    return samples


def summarize(samples: list[Sample]) -> dict[str, dict[str, tuple[float, float]]]:
    """Mean and standard deviation of each feature, per species."""
    by_species: dict[str, list[Sample]] = {}
    for s in samples:
        by_species.setdefault(s.species, []).append(s)
    result = {}
    for species, group in sorted(by_species.items()):
        stats = {}
        for i, name in enumerate(FEATURES):
            column = [s.features[i] for s in group]
            sd = stdev(column) if len(column) > 1 else 0.0
            stats[name] = (round(mean(column), 3), round(sd, 3))
        result[species] = stats
    return result
