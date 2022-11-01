import os

import srsly
import typer


from src.data import load_data, sample_fold

app = typer.Typer()


@app.command()
def create_data(
    data_dir="data",
    dataset="imdb",
    n_shot: int = 8,
    test_size: int = 100,
    split="train",
):
    data = load_data(dataset, test_size)

    num_classes = len(set(data["train"]["label"]))
    sample_size = num_classes * n_shot

    if split == "train":
        data = sample_fold(data["train"], 1, sample_size)
    else:
        data = data["test"]

    data_path = os.path.join(data_dir, f"{dataset}_{n_shot}shot_{split}.jsonl")
    srsly.write_jsonl(data_path, data)


if __name__ == "__main__":
    app()
