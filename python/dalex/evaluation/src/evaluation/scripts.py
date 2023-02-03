import time
from collections import defaultdict
from itertools import product
from typing import Optional, Tuple

import click
import dalex as dx
import pandas as pd
import shap
from evaluation.config import DATA_DIR, DATASETS, METHODS, MODELS
from evaluation.data import load_dataset
from evaluation.experiment import run_experiment
from evaluation.models import create_model
from sklearn.model_selection import train_test_split


def evaluation():
    ...


def run_explanation_experiments(
    model: str,
    method: str,
    dataset: str,
    n_runs: int,
    observation_id: int,
    n_samples: Optional[int] = None,
    save: bool = True,
) -> Optional[Tuple[float, pd.DataFrame]]:
    """run experiments, get results and runtime and either return time (of explanations only)
    and the dataframe or save df it in `DATA_DIR` folder"""

    X, y = load_dataset(dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=300, shuffle=True, random_state=446519
    )

    my_model = create_model(model)
    my_model.fit(X_train, y_train)

    if method not in METHODS:
        raise ValueError
    elif method == "exact":
        explainer = shap.explainers.Exact(my_model.predict, X_train)
    elif method == "kernel":
        explainer = shap.KernelExplainer(my_model.predict, X_train)
    elif method == "unbiased":
        explainer = dx.Explainer(
            my_model, X_train, y_train, predict_function=lambda m, d: m.predict(d)
        )

    result_columns = list(X_train.columns) + [
        "run_idx",
        "model",
        "method",
        "dataset",
        "observation_id",
        "n_samples",
    ]
    shap_results = pd.DataFrame(columns=result_columns)

    measured_time = 0

    for run_idx in range(0, n_runs):
        print(run_idx, model, method, dataset, n_samples, observation_id)
        if method == "unbiased":
            start = time.time()
            explain = explainer.predict_parts(
                X_test.iloc[observation_id : observation_id + 1],
                type="unbiased_kernel_shap",
                n_samples=n_samples,
            )
            measured_time += time.time() - start
            shap_values = explain.result["contribution"]
        elif method == "kernel":
            start = time.time()
            shap_values = explainer.shap_values(
                X_test.iloc[observation_id : observation_id + 1], nsamples=n_samples
            )
            measured_time += time.time() - start
        elif method == "exact":
            start = time.time()
            shap_values = explainer(
                X_test.iloc[observation_id : observation_id + 1]
            ).values
            measured_time += time.time() - start

        row = shap_values.tolist().append(
            [run_idx, model, method, dataset, observation_id, n_samples]
        )
        temp_df = pd.DataFrame(row, columns=result_columns)

        shap_results = shap_results.append(temp_df, ignore_index=True)

    if save:
        shap_results.to_parquet(
            DATA_DIR
            / "estimates"
            / f"model_{model}_meathod_{method}_dataset_{dataset}_nruns_{n_runs}_observation_id_{observation_id}_nsmaples_{n_samples}.parquet"
        )

    return measured_time, shap_results


@click.command()
@click.option("--models", type=str, required=True, help="comma separated values pls")
@click.option("--methods", type=str, required=True, help="comma separated values pls")
@click.option("--datasets", type=str, required=True, help="comma separated values pls")
@click.option("--n_runs", type=int, required=True, help="comma separated values pls")
@click.option(
    "--observation_ids", type=str, required=True, help="comma separated values pls"
)
@click.option(
    "--n_samples", type=str, required=False, help="comma separated values pls"
)
def run_many_experiments(
    models: str,
    methods: str,
    datasets: str,
    n_runs: int,
    observation_ids: str,
    n_samples: Optional[str],
) -> None:
    models_p = models.split(",")
    methods_p = methods.split(",")
    datasets_p = datasets.split(",")
    observation_ids_p = [int(v) for v in observation_ids.split(",")]
    n_samples_p = [int(v) for v in n_samples.split(",")] if n_samples else [None]

    results = defaultdict(list)
    times = []
    for model, method, dataset, n_sample, observation_id in product(
        models_p, methods_p, datasets_p, n_samples_p, observation_ids_p
    ):
        m_time, df = run_explanation_experiments(
            model,
            method,
            dataset,
            n_runs,
            observation_id,
            n_sample if method != "exact" else None,
            save=False,
        )
        times.append([model, method, dataset, n_runs, observation_id, n_sample, m_time])
        results[dataset].append(df)

    for name, dfs in results.items():
        df = pd.concat(dfs)
        df.to_parquet(
            DATA_DIR
            / "estimates"
            / f"{dataset}_samples{observation_ids.replace(',','_')}_all.parquet"
        )
    pd.DataFrame(
        data=times,
        columns=[
            "model",
            "method",
            "dataset",
            "n_runs",
            "observation_ids",
            "n_sample",
            "time",
        ],
    ).to_parquet(
        DATA_DIR
        / "estimates"
        / f"{dataset}_samples{observation_ids.replace(',','_')}_times_all.parquet"
    )


if __name__ == "__main__":
    run_many_experiments()
