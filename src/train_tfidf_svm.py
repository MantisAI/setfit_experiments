from pathlib import Path
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from datasets import load_dataset
import typer

from src.data import load_data, sample_fold, convert_data

app = typer.Typer()


@app.command()
def train_tfidf_svm(
    data_path="imdb", n_shot: int = 8, n_folds: int = 5, results_dir="results"
):
    dataset = load_data(data_path)

    num_classes = len(set(dataset["train"]["label"]))
    sample_size = num_classes * n_shot

    results = []
    for fold in range(n_folds):
        train_dataset = sample_fold(dataset, fold, sample_size)

        X_train, y_train = convert_data(train_dataset)
        X_test, y_test = convert_data(dataset["test"])

        model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(min_df=5, max_features=10_000)),
                ("svm", SGDClassifier()),
            ]
        )
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)
        print(f"Score: {score}")

        results.append({"fold": fold, "metrics": {"accuracy": score}})

    results_path = os.path.join(results_dir, data_path, "tfidf_svm")
    Path(results_path).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(results_path, f"{n_shot}.json"), "w") as f:
        f.write(json.dumps(results))


if __name__ == "__main__":
    app()
