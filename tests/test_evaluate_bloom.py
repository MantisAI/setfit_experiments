from typer.testing import CliRunner
from src.evaluate_bloom import app

runner = CliRunner()


def test_evaluate_gpt3(tmp_path, monkeypatch):
    monkeypatch.setattr("src.evaluate_bloom.bloom", lambda model, prompt: "0")
    result = runner.invoke(app, ["--data-path", "ag_news", "--results-dir", tmp_path])
    assert result.exit_code == 0
