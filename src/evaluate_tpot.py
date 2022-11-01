from sklearn.feature_extraction.text import TfidfVectorizer
from tpot import TPOTClassifier
import numpy as np
import typer

from data import load_data, sample_fold, convert_data

app = typer.Typer()


@app.command()
def evaluate_tpot(data_path="ag_news", n_shot: int = 8, test_size: int = 100):
    dataset = load_data(data_path, test_size)

    num_classes = len(set(dataset["train"]["label"]))
    sample_size = num_classes * n_shot

    train_dataset = sample_fold(dataset["train"], 1, sample_size)

    X_train, y_train = convert_data(train_dataset)
    X_test, y_test = convert_data(dataset["test"])

    tfidf = TfidfVectorizer(min_df=5, stop_words="english", max_features=10_000)
    tfidf.fit(X_train)

    X_train = tfidf.transform(X_train)
    X_test = tfidf.transform(X_test)

    tpot = TPOTClassifier(
        generations=5,
        population_size=20,
        cv=5,
        random_state=42,
        verbosity=2,
        config_dict="TPOT sparse",
    )

    tpot.fit(X_train, y_train)

    score = tpot.score(X_test, np.array(y_test))
    print(score)


if __name__ == "__main__":
    app()
