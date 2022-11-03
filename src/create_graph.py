import json
import os

import seaborn as sns
import pandas as pd
import typer

app = typer.Typer()


@app.command()
def create_graph(results_path="results", figures_path="figures"):
    for dataset in os.listdir(results_path):

        all_data = []
        few_shot_data = []
        dataset_results_path = os.path.join(results_path, dataset)
        for model in os.listdir(dataset_results_path):

            model_results_path = os.path.join(dataset_results_path, model)
            for result_filename in os.listdir(model_results_path):

                result_path = os.path.join(model_results_path, result_filename)
                with open(result_path) as f:
                    result = json.loads(f.read())

                if result_filename == "all.json":
                    all_data.append(
                        {
                            "model": f"{model} - (all data)",
                            "accuracy": result[0]["metrics"]["accuracy"],
                        }
                    )
                else:
                    n_shot, _ = os.path.splitext(result_filename)

                    for fold_result in result:
                        few_shot_data.append(
                            {
                                "model": model,
                                "n_shot": int(n_shot),
                                "fold": fold_result["fold"],
                                "accuracy": fold_result["metrics"]["accuracy"],
                            }
                        )

        few_shot_data = pd.DataFrame(few_shot_data)
        line_plot = sns.lineplot(few_shot_data, x="n_shot", y="accuracy", hue="model")

        for result in all_data:
            line_plot.axhline(result["accuracy"], linestyle="--", label=result["model"])

        handles, labels = line_plot.get_legend_handles_labels()
        line_plot.legend(handles, labels)

        fig = line_plot.get_figure()
        figure_path = os.path.join(figures_path, f"{dataset}.png")
        fig.savefig(figure_path)


if __name__ == "__main__":
    app()
