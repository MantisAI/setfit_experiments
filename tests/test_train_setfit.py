from typer.testing import CliRunner
from src.train_setfit import app

runner = CliRunner()


def test_train_set_fit(tmp_path):
    result = runner.invoke(
        app,
        [
            "--data-path",
            "ag_news",
            "--model-path",
            "sentence-transformers/all-MiniLM-L6-v2",
            "--n-shot",
            "1",
            "--test-size",
            "5",
            "--n-folds",
            "2",
            "--results-dir",
            tmp_path,
        ],
    )
    assert result.exit_code == 0
