from pathlib import Path
import json
import os

import typer


def record_manual(dataset, approach, nshot, results, results_dir="results"):
    result_data = []
    for fold, result in enumerate(results.split(",")):
        result_data.append({"fold": fold, "metrics": {"accuracy": float(result)}})

    results_path = os.path.join(results_dir, dataset, approach)
    Path(results_path).mkdir(exist_ok=True, parents=True)

    with open(os.path.join(results_path, f"{nshot}.json"), "w") as f:
        f.write(json.dumps(result_data))


if __name__ == "__main__":
    typer.run(record_manual)
