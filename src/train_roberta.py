from pathlib import Path
import json
import os

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset, Dataset
import numpy as np
import evaluate
import typer

from src.data import load_data, sample_data

app = typer.Typer()


@app.command()
def train(
    data_path="imdb",
    model_path="roberta-base",
    n_shot: str = "8",
    epochs: int = 5,
    batch_size: int = 16,
    weight_decay: float = 0.01,
    test_size: int = 100,
    n_folds: int = 5,
    results_dir="results",
    models_dir="models",
):
    dataset = load_data(data_path)

    results = []
    for fold in range(n_folds):
        if n_shot == "all":
            sample_dataset = dataset
        else:
            n_shot = int(n_shot)
            sample_dataset = sample_data(
                dataset, num_samples_per_class=n_shot, test_size=test_size, seed=fold
            )

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        def tokenize(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        sample_dataset = sample_dataset.map(tokenize, batched=True)

        num_labels = len(set(dataset["train"]["label"]))
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels
        )

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        training_args = TrainingArguments(
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            output_dir=f"{models_dir}/{data_path}/roberta",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=sample_dataset["train"],
            eval_dataset=sample_dataset["test"],
            compute_metrics=compute_metrics,
        )
        trainer.train()

        metrics = trainer.evaluate()
        print(f"fold: {fold} - metrics {metrics}")

        results.append(
            {"fold": fold, "metrics": {"accuracy": metrics["eval_accuracy"]}}
        )

    results_path = os.path.join(results_dir, data_path, "roberta")
    Path(results_path).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(results_path, f"{n_shot}.json"), "w") as f:
        f.write(json.dumps(results))


if __name__ == "__main__":
    app()
