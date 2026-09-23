import pytest

from iris_explorer.data import load
from iris_explorer.knn import KNN, accuracy, confusion_matrix, cross_validate


def test_simple_clusters():
    x = [(0.0, 0.0), (0.1, 0.2), (5.0, 5.0), (5.2, 4.9)]
    y = ["a", "a", "b", "b"]
    model = KNN(k=1).fit(x, y)
    assert model.predict([(0.05, 0.1), (4.9, 5.1)]) == ["a", "b"]


def test_tie_goes_to_closest():
    model = KNN(k=2).fit([(0.0,), (1.0,)], ["near", "far"])
    assert model.predict_one((0.2,)) == "near"


def test_invalid_k_and_unfitted():
    with pytest.raises(ValueError):
        KNN(k=0)
    with pytest.raises(RuntimeError):
        KNN().predict_one((1.0,))


def test_confusion_matrix_counts():
    labels, m = confusion_matrix(["a", "a", "b"], ["a", "b", "b"])
    assert labels == ["a", "b"]
    assert m == [[1, 1], [0, 1]]
    assert accuracy(["a", "a", "b"], ["a", "b", "b"]) == pytest.approx(2 / 3)


def test_iris_accuracy_is_high():
    samples = load()
    x = [s.features for s in samples]
    y = [s.species for s in samples]
    assert cross_validate(x, y, k=5) > 0.9
