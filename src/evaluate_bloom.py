from pathlib import Path
import json
import os

from sklearn.metrics import accuracy_score
from tqdm import tqdm
import requests
import typer

from src.data import load_data, sample_fold

app = typer.Typer()

API_URL = "https://api-inference.huggingface.co/models"


def create_prompt(dataset):
    labels = list(set([str(example["label"]) for example in dataset]))

    prompt = "Classify the sentence as one of {}\n".format(",".join(labels))

    for example in dataset:
        text, label = example["text"], example["label"]
        prompt += f"Text:{text}\n"
        prompt += f"Label:{label}\n\n"

    return prompt


def bloom(model, prompt):
    api_key = os.getenv("HF_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.post(f"{API_URL}/{model}", headers=headers, data=prompt)
    response = json.loads(response.content.decode("utf-8"))
    return response["data"][0]["generated_text"].replace(prompt, "")


@app.command()
def evaluate(
    data_path="imdb",
    model="bigscience/bloom",
    n_shot: int = 8,
    n_folds: int = 5,
    test_size: int = 100,
    results_dir="results",
):
    dataset = load_data(data_path, test_size)

    num_classes = len(set(dataset["train"]["label"]))
    sample_size = num_classes * n_shot
    print(f"Sample size: {sample_size}")

    train_dataset = sample_fold(dataset["train"], 1, sample_size)
    test_dataset = dataset["test"]

    prompt = create_prompt(train_dataset)

    results = []
    for fold in range(n_folds):
        y_pred = []
        for example in tqdm(test_dataset):
            text = example["text"]

            prompt += f"Text:{text}\nLabel:"

            pred = bloom(model, prompt)

            y_pred.append(pred)

        y_test = [str(example["label"]) for example in test_dataset]

        score = accuracy_score(y_test, y_pred)

        print(y_test)
        print(y_pred)
        print(score)

        results.append({"fold": fold, "metrics": {"accuracy": score}})

    results_path = os.path.join(results_dir, data_path, "bloom")
    Path(results_path).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(results_path, f"{n_shot}.json"), "w") as f:
        f.write(json.dumps(results))


if __name__ == "__main__":
    app()
