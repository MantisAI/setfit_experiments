from pathlib import Path
import os

import srsly
import typer


from src.data import load_data, sample_data

app = typer.Typer()


@app.command()
def create_data(
    data_dir="data",
    dataset="imdb",
    n_shot: int = 8,
    n_folds: int = 5,
    test_size: int = 100,
    split="train",
):
    data = load_data(dataset)

    data_dir = os.path.join(data_dir, dataset, str(n_shot))
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    for fold in range(n_folds):
        sample_dataset = sample_data(
            data, num_samples_per_class=n_shot, test_size=test_size, seed=fold
        )

        data_path = os.path.join(data_dir, f"fold{fold}_{split}.jsonl")
        srsly.write_jsonl(data_path, sample_dataset[split])


if __name__ == "__main__":
    app()
