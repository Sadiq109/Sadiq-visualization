"""Command line entry point: python -m iris_explorer [--plots DIR]."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import FEATURES, load, summarize
from .knn import KNN, accuracy, confusion_matrix, cross_validate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explore and classify the Iris dataset.")
    parser.add_argument("--data", type=Path, help="path to iris.csv")
    parser.add_argument("--plots", type=Path, help="write figures to this folder")
    parser.add_argument("--max-k", type=int, default=15)
    args = parser.parse_args(argv)

    samples = load(args.data) if args.data else load()
    x = [s.features for s in samples]
    y = [s.species for s in samples]

    print(f"{len(samples)} samples\n")
    print("Mean (sd) per species")
    for species, stats in summarize(samples).items():
        cells = ", ".join(f"{f}={m} ({sd})" for f, (m, sd) in stats.items())
        print(f"  {species}: {cells}")

    results = {k: cross_validate(x, y, k) for k in range(1, args.max_k + 1, 2)}
    best_k = max(results, key=lambda k: (results[k], -k))
    print("\n5-fold accuracy by k")
    for k, score in results.items():
        print(f"  k={k:<2} {score:.3f}")
    print(f"\nBest k: {best_k}")

    # Hold out every 5th sample for a final check with the chosen k.
    test = set(range(0, len(x), 5))
    train = [i for i in range(len(x)) if i not in test]
    model = KNN(best_k).fit([x[i] for i in train], [y[i] for i in train])
    truth = [y[i] for i in sorted(test)]
    pred = model.predict([x[i] for i in sorted(test)])
    labels, matrix = confusion_matrix(truth, pred)
    print(f"Hold-out accuracy: {accuracy(truth, pred):.3f}")
    print("Confusion matrix (rows = actual)")
    width = max(len(l) for l in labels)
    print(" " * (width + 3) + " ".join(f"{l[:4]:>5}" for l in labels))
    for label, row in zip(labels, matrix):
        print(f"  {label:<{width}} " + " ".join(f"{n:>5}" for n in row))

    if args.plots:
        from .plots import k_curve, pair_grid

        args.plots.mkdir(parents=True, exist_ok=True)
        pair_grid(samples, args.plots / "pair_grid.png")
        k_curve(results, args.plots / "k_curve.png")
        print(f"\nSaved figures to {args.plots}/")
    return 0
