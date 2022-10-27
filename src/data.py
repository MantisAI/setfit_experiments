import json

from datasets import load_dataset, Dataset


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


def load_data(data_path, test_size):
    if "jsonl" in data_path:
        data = read_jsonl(data_path)
        dataset = Dataset.from_dict(data)
        dataset = dataset.train_test_split(test_size=test_size)
    else:
        dataset = load_dataset(data_path)
        dataset["test"] = dataset["test"].select(range(test_size))

    return dataset


def convert_data(data):
    X, y = zip(*[(example["text"], example["label"]) for example in data])
    return X, y


def sample_fold(dataset, fold_n, sample_size):
    return dataset.shuffle(seed=fold_n).select(range(sample_size))
