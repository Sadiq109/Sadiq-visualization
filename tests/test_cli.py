from iris_explorer.cli import main


def test_cli_runs_and_writes_plots(tmp_path, capsys):
    assert main(["--plots", str(tmp_path), "--max-k", "5"]) == 0
    out = capsys.readouterr().out
    assert "Best k" in out and "Confusion matrix" in out
    assert (tmp_path / "pair_grid.png").exists()
    assert (tmp_path / "k_curve.png").exists()
