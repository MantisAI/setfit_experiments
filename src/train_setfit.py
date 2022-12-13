from pathlib import Path
import json
import os

from setfit import SetFitModel, SetFitTrainer
from datasets import load_dataset, Dataset
import typer

from src.data import load_data, sample_data

app = typer.Typer()


@app.command()
def train(
    data_path="imdb",
    model_path="sentence-transformers/paraphrase-mpnet-base-v2",
    n_shot: int = 8,
    num_iterations: int = 5,
    epochs: int = 1,
    batch_size: int = 16,
    test_size: int = 100,
    n_folds: int = 5,
    results_dir="results",
):
    dataset = load_data(data_path)

    results = []
    for fold in range(n_folds):
        sample_dataset = sample_data(
            dataset, num_samples_per_class=n_shot, test_size=test_size, seed=fold
        )

        model = SetFitModel.from_pretrained(model_path)

        trainer = SetFitTrainer(
            model=model,
            train_dataset=sample_dataset["train"],
            eval_dataset=sample_dataset["test"],
            num_iterations=num_iterations,
            num_epochs=epochs,
            batch_size=batch_size,
            metric="accuracy",
            column_mapping={"text": "text", "label": "label"},
        )
        trainer.train()

        metrics = trainer.evaluate()
        print(f"fold: {fold} - metrics {metrics}")

        results.append({"fold": fold, "metrics": metrics})

    if "/" in data_path:
        # if dataset comes from custom account, keep just the name
        data_path = data_path.split("/")[1]
    results_path = os.path.join(results_dir, data_path, "setfit")
    Path(results_path).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(results_path, f"{n_shot}.json"), "w") as f:
        f.write(json.dumps(results))


if __name__ == "__main__":
    app()
