from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from datasets import load_dataset
import typer


def convert_data(data):
    X, y = zip(*[(example["text"], example["label"]) for example in data])
    return X, y


def train_tfidf_svm(data_path="imdb", n_shot: int = 8):
    dataset = load_dataset(data_path)

    num_classes = len(set(dataset["train"]["label"]))
    train_dataset = dataset["train"].shuffle(42).select(range(n_shot))

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


if __name__ == "__main__":
    typer.run(train_tfidf_svm)
