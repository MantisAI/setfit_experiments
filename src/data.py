import random
import json

from datasets import load_dataset, Dataset, ClassLabel


def read_jsonl(data_path):
    texts = []
    labels = []
    with open(data_path) as f:
        for line in f:
            example = json.loads(line)
            texts.append(example["text"])
            labels.append(example["intent"])

    label2id = {label: idx for idx, label in enumerate(set(labels))}
    labels = [label2id[label] for label in labels]
    return {"text": texts, "label": labels}


def load_data(data_path):
    if "jsonl" in data_path:
        data = read_jsonl(data_path)
        dataset = Dataset.from_dict(data)
    else:
        if data_path == "tweet_eval":
            dataset = load_dataset(data_path, "emotion")
        else:
            dataset = load_dataset(data_path)

    if data_path == "trec":
        dataset = dataset.rename_column("label-coarse", "label")
    if data_path == "SetFit/enron_spam":
        label = ClassLabel(num_classes=2, names=["ham", "spam"])
        dataset = dataset.cast_column("label", label)
    if data_path == "SetFit/toxic_conversations":
        label = ClassLabel(num_classes=2, names=["non toxic", "toxic"])
        dataset = dataset.cast_column("label", label)
    return dataset


def sample_data(dataset, num_samples_per_class, test_size, seed=42):
    random.seed(seed)
    labels = list(set(dataset["train"]["label"]))

    dataset = dataset["train"].train_test_split(
        test_size=test_size, stratify_by_column="label", seed=seed
    )

    train_indices = []
    for label in labels:
        label_indices = [
            i for i, example in enumerate(dataset["train"]) if example["label"] == label
        ]
        random.shuffle(label_indices)
        label_indices = label_indices[:num_samples_per_class]
        train_indices.extend(label_indices)
    dataset["train"] = dataset["train"].select(train_indices)

    return dataset


def convert_data(data):
    X, y = zip(*[(example["text"], example["label"]) for example in data])
    return X, y
