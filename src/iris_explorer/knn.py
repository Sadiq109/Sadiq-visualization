"""A k-nearest-neighbors classifier written from scratch.

I wrote this without scikit-learn so I could see how the algorithm works:
standardize the features, find the k closest training points by Euclidean
distance and take a majority vote (ties go to the closest neighbor's class).
"""

from __future__ import annotations

import math
import random
from collections import Counter
from statistics import mean, pstdev

Vector = tuple[float, ...]


class KNN:
    def __init__(self, k: int = 5) -> None:
        if k < 1:
            raise ValueError("k must be at least 1")
        self.k = k
        self._x: list[Vector] = []
        self._y: list[str] = []
        self._means: list[float] = []
        self._stds: list[float] = []

    def fit(self, x: list[Vector], y: list[str]) -> "KNN":
        if len(x) != len(y) or not x:
            raise ValueError("x and y must be non-empty and the same length")
        cols = list(zip(*x))
        self._means = [mean(c) for c in cols]
        self._stds = [pstdev(c) or 1.0 for c in cols]
        self._x = [self._scale(v) for v in x]
        self._y = list(y)
        return self

    def _scale(self, v: Vector) -> Vector:
        return tuple((a - m) / s for a, m, s in zip(v, self._means, self._stds))

    def predict_one(self, v: Vector) -> str:
        if not self._x:
            raise RuntimeError("call fit() first")
        q = self._scale(v)
        nearest = sorted(
            range(len(self._x)), key=lambda i: math.dist(q, self._x[i])
        )[: self.k]
        votes = Counter(self._y[i] for i in nearest)
        top = max(votes.values())
        for i in nearest:  # closest neighbor wins a tie
            if votes[self._y[i]] == top:
                return self._y[i]
        raise AssertionError("unreachable")

    def predict(self, xs: list[Vector]) -> list[str]:
        return [self.predict_one(v) for v in xs]


def accuracy(truth: list[str], predicted: list[str]) -> float:
    if len(truth) != len(predicted) or not truth:
        raise ValueError("lists must be non-empty and the same length")
    return sum(a == b for a, b in zip(truth, predicted)) / len(truth)


def confusion_matrix(truth: list[str], predicted: list[str]) -> tuple[list[str], list[list[int]]]:
    labels = sorted(set(truth) | set(predicted))
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(truth, predicted):
        matrix[index[t]][index[p]] += 1
    return labels, matrix


def cross_validate(x: list[Vector], y: list[str], k: int, folds: int = 5, seed: int = 0) -> float:
    """Mean accuracy over shuffled folds."""
    order = list(range(len(x)))
    random.Random(seed).shuffle(order)
    scores = []
    for f in range(folds):
        test = set(order[f::folds])
        train = [i for i in order if i not in test]
        model = KNN(k).fit([x[i] for i in train], [y[i] for i in train])
        scores.append(accuracy([y[i] for i in test], model.predict([x[i] for i in test])))
    return sum(scores) / len(scores)
