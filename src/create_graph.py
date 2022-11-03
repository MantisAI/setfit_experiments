import json
import os

import seaborn as sns
import pandas as pd
import typer

app = typer.Typer()


@app.command()
def create_graph(results_path="results", figures_path="figures"):
    for dataset in os.listdir(results_path):

        data = []
        dataset_results_path = os.path.join(results_path, dataset)
        for model in os.listdir(dataset_results_path):

            model_results_path = os.path.join(dataset_results_path, model)
            for result_filename in os.listdir(model_results_path):
                n_shot, _ = os.path.splitext(result_filename)

                result_path = os.path.join(model_results_path, result_filename)
                with open(result_path) as f:
                    result = json.loads(f.read())

                for fold_result in result:
                    data.append(
                        {
                            "model": model,
                            "n_shot": int(n_shot),
                            "fold": fold_result["fold"],
                            "accuracy": fold_result["metrics"]["accuracy"],
                        }
                    )

        data = pd.DataFrame(data)
        line_plot = sns.lineplot(data, x="n_shot", y="accuracy", hue="model")

        fig = line_plot.get_figure()
        figure_path = os.path.join(figures_path, f"{dataset}.png")
        fig.savefig(figure_path)


if __name__ == "__main__":
    app()
