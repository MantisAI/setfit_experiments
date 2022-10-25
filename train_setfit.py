from setfit import SetFitModel, SetFitTrainer
from datasets import load_dataset
import typer


def train(
    data_path="imdb",
    model_path="sentence-transformers/paraphrase-mpnet-base-v2",
    n_shot: int = 8,
    num_iterations: int = 20,
    epochs: int = 1,
    batch_size: int = 16,
    test_size: int = 100,
):
    dataset = load_dataset("imdb")

    num_classes = len(set(dataset["train"]["label"]))
    train_dataset = (
        dataset["train"].shuffle(seed=42).select(range(n_shot * num_classes))
    )
    eval_dataset = dataset["test"].select(range(test_size))

    model = SetFitModel.from_pretrained(model_path)

    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_iterations=num_iterations,
        num_epochs=epochs,
        batch_size=batch_size,
        # loss_class
        # metric
        column_mapping={"text": "text", "label": "label"},
    )
    trainer.train()

    metrics = trainer.evaluate()
    print(metrics)


if __name__ == "__main__":
    typer.run(train)
