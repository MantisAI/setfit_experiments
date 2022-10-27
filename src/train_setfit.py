from pathlib import Path
import json
import os

from setfit import SetFitModel, SetFitTrainer
from datasets import load_dataset, Dataset
import typer

from src.data import load_data, sample_fold

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
    dataset = load_data(data_path, test_size)

    num_classes = len(set(dataset["train"]["label"]))
    sample_size = num_classes * n_shot

    results = []
    for fold in range(n_folds):
        train_dataset = sample_fold(dataset["train"], fold, sample_size)
        eval_dataset = dataset["test"]

        model = SetFitModel.from_pretrained(model_path)

        trainer = SetFitTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
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

    results_path = os.path.join(results_dir, data_path, "setfit")
    Path(results_path).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(results_path, f"{n_shot}shot.json"), "w") as f:
        f.write(json.dumps(results))


if __name__ == "__main__":
    app()
