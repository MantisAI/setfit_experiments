import json

from datasets import load_dataset, Dataset


LABEL_MAP = {"ag_news": {0: "world", 1: "sports", 2: "business", 3: "sci/tech"}}


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
        dataset = load_dataset(data_path)

    return dataset


def sample_data(dataset, num_samples_per_class, test_size, seed=42):
    num_classes = len(set(dataset["train"]["label"]))
    train_size = num_classes * num_samples_per_class

    dataset = dataset["train"].train_test_split(
        train_size=train_size, stratify_by_column="label", seed=seed
    )
    dataset["test"] = dataset["test"].select(range(test_size))

    return dataset


def convert_data(data):
    X, y = zip(*[(example["text"], example["label"]) for example in data])
    return X, y
