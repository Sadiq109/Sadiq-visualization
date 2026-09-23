import pytest

from iris_explorer.data import load, summarize


def test_load_full_dataset():
    samples = load()
    assert len(samples) == 150
    assert {s.species for s in samples} == {"setosa", "versicolor", "virginica"}


def test_summary_known_value():
    stats = summarize(load())
    mean_petal, _ = stats["setosa"]["petal_length"]
    assert mean_petal == pytest.approx(1.462, abs=1e-3)


def test_bad_row_reports_line(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("sepal_length,sepal_width,petal_length,petal_width,species\n5.1,x,1.4,0.2,setosa\n")
    with pytest.raises(ValueError, match="line 2"):
        load(p)


def test_missing_column(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="missing columns"):
        load(p)
