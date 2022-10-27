from typer.testing import CliRunner
from src.evaluate_gpt3 import app

runner = CliRunner()


def test_evaluate_gpt3(tmp_path, monkeypatch):
    monkeypatch.setattr("src.evaluate_gpt3.gpt3", lambda model, prompt: "0")
    result = runner.invoke(app, ["--results-dir", tmp_path])
    assert result.exit_code == 0
