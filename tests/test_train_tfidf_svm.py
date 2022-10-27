from typer.testing import CliRunner
from src.train_tfidf_svm import app

runner = CliRunner()


def test_train_tfidf_svm(tmp_path):
    result = runner.invoke(
        app,
        [
            "--data-path",
            "ag_news",
            "--n-folds",
            "3",
            "--test-size",
            "5",
            "--results-dir",
            tmp_path,
        ],
    )
    assert result.exit_code == 0
