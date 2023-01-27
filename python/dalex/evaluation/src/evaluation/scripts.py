from collections import defaultdict
from itertools import product
from typing import Optional, Tuple

import click
import pandas as pd

from evaluation.config import DATA_DIR, DATASETS, METHODS, MODELS
from evaluation.experiment import run_experiment


@click.group
def evaluation():
    ...


@evaluation.command()
@click.option("--model", type=click.Choice(MODELS), required=True)
@click.option("--method", type=click.Choice(METHODS), required=True)
@click.option("--dataset", type=click.Choice(DATASETS), required=True)
@click.option("--n_runs", type=int, required=True)
@click.option("--n_samples", type=int, required=False)
def run_explanation_experiments(
    model: str,
    method: str,
    dataset: str,
    n_runs: int,
    n_samples: Optional[int] = None,
    save: bool = True,
) -> Optional[Tuple[float, pd.DataFrame]]:
    """run experiments, get results and runtime and either return time (of explanations only)
    and the dataframe or save df it in `DATA_DIR` folder"""
    ...  # TODO


@evaluation.command()
@click.option("--models", type=str, required=True, help="comma separated values pls")
@click.option("--methods", type=str, required=True, help="comma separated values pls")
@click.option("--datasets", type=str, required=True, help="comma separated values pls")
@click.option("--n_runs", type=str, required=True, help="comma separated values pls")
@click.option(
    "--n_samples", type=str, required=False, help="comma separated values pls"
)
def run_many_experiments(
    models: str,
    methods: str,
    datasets: str,
    n_runs: str,
    n_samples: Optional[str],
) -> None:
    models_p = models.split(",")
    methods_p = methods.split(",")
    datasets_p = datasets.split(",")
    n_runs_p = [int(v) for v in n_runs.split(",")]
    n_samples_p = [int(v) for v in n_samples.split(",")] if n_samples else [None]

    results = defaultdict(list)
    times = []
    for model, method, dataset, n_run, n_sample in product(
        models_p, methods_p, datasets_p, n_runs_p, n_samples_p
    ):
        time, df = run_explanation_experiments(
            model,
            method,
            dataset,
            n_run,
            n_sample if model != "kernel" else None,
            save=False,
        )
        times.append([model, method, dataset, n_run, n_sample, time])
        results[dataset].append(df)

    for name, dfs in results.items():
        df = pd.concat(dfs)
        df.to_parquet(DATA_DIR / "estimates" / f"{name}_all.parquet")
    pd.DataFrame(
        data=times, columns=["model", "method", "dataset", "n_run", "n_sample", "time"]
    ).to_parquet(DATA_DIR / "estimates" / f"run_times_all.parquet")


if __name__ == "__main__":
    evaluation()
