"""Explain data already held in Polars: native versus one-time pandas conversion.

Model fitting and creation of the original Polars frame are outside the timed
region. The pandas route includes to_pandas; the native route uses a cheap clone.
Workers are isolated, alternate backend order and warm their own runtime.
"""
import argparse
import gc
import importlib.metadata as metadata
import json
import multiprocessing as mp
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

import dalex as dx
from _benchmark_utils import RSS, summary, source_identity


def explain(exp, family, sample, rounds):
    obs = exp.data.head(1)
    if family == "importance":
        return exp.model_parts(N=sample, B=rounds, random_state=42).result
    if family in ("breakdown", "shap"):
        return exp.predict_parts(obs, type="break_down" if family == "breakdown" else "shap",
                                 N=sample, B=rounds, random_state=42).result
    return exp.model_profile(type=family, N=sample, variables=["x0", "x1"],
                             grid_points=10, random_state=42, verbose=False).result


def worker(queue, backend, values, y, model, family, sample, rounds):
    try:
        with threadpool_limits(limits=1):
            original = pl.DataFrame(values, schema=[f"x{i}" for i in range(values.shape[1])])
            for warm in (True, False):
                with RSS() as rss:
                    start = perf_counter()
                    data = original.to_pandas() if backend == "pandas" else original.clone()
                    converted = perf_counter()
                    exp = dx.Explainer(model, data, y, model_type="regression", verbose=False,
                                       precalculate=False, label="model")
                    initialized = perf_counter()
                    result = explain(exp, family, sample, rounds)
                    finished = perf_counter()
                if not warm:
                    queue.put({"backend": backend, "seconds": finished-start,
                               "conversion_seconds": converted-start,
                               "initialization_seconds": initialized-converted,
                               "explanation_seconds": finished-initialized,
                               "peak_rss_bytes": rss.peak,
                               "rss_increase_bytes": max(0, rss.peak-rss.baseline),
                               "outcome": result.select_dtypes(include=np.number).to_numpy(dtype=float).tolist()})
                else:
                    del data, exp, result
                    gc.collect()
    except BaseException:
        queue.put({"error": traceback.format_exc()})


def main():
    import hashlib
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["sklearn", "lightgbm"], required=True)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--features", type=int, default=10)
    parser.add_argument("--model-file", help="Reuse one fitted model across source revisions; create if absent")
    parser.add_argument("--source-revision", default="unspecified")
    parser.add_argument("--backends", nargs="+", choices=["pandas", "polars"], default=["pandas", "polars"],
                        help="Use pandas alone to benchmark an unmodified upstream source")
    parser.add_argument("--sample", type=int, default=1000)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--families", nargs="+", default=["importance", "breakdown", "shap", "partial", "accumulated", "conditional"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(123)
    values = rng.normal(size=(args.rows, args.features))
    original = pl.DataFrame(values, schema=[f"x{i}" for i in range(args.features)])
    y = values[:, :5].sum(axis=1) + rng.normal(size=args.rows)*.1
    with threadpool_limits(limits=1):
        if args.model_file and Path(args.model_file).exists():
            model = pickle.loads(Path(args.model_file).read_bytes())
            if model.n_features_in_ != args.features:
                raise ValueError("cached model has a different feature count")
        elif args.model == "lightgbm":
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(n_estimators=20, n_jobs=1, verbosity=-1, random_state=42).fit(original, y)
        else:
            model = HistGradientBoostingRegressor(max_iter=20, random_state=42).fit(original, y)
    if args.model_file and not Path(args.model_file).exists():
        Path(args.model_file).write_bytes(pickle.dumps(model))
    packages = ["dalex", "pandas", "polars", "narwhals", "numpy", "scikit-learn"]
    if args.model == "lightgbm":
        packages.append("lightgbm")
    report = {"python": sys.version, "platform": platform.platform(), "settings": vars(args),
              "polars_thread_pool_size": pl.thread_pool_size(),
              "model_and_blas_threads": 1,
              "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: None)(),
              "source_revision": args.source_revision,
              "library_source": source_identity(dx),
              "fitted_model_sha256": hashlib.sha256(Path(args.model_file).read_bytes()).hexdigest() if args.model_file else None,
              "versions": {name: metadata.version(name) for name in packages},
              "input_origin": "Polars; original creation/training excluded; pandas conversion included",
              "runtime": "warm inside each measured worker", "families": {}}
    context = mp.get_context("spawn")
    for family in args.families:
        trials, expected, error = [], None, 0.
        for repeat in range(args.repetitions):
            for backend in (args.backends if repeat % 2 == 0 else args.backends[::-1]):
                queue = context.Queue()
                process = context.Process(target=worker, args=(queue,backend,values,y,model,family,min(args.sample,args.rows),args.rounds))
                process.start()
                item = queue.get(timeout=600)
                process.join(timeout=30)
                if process.exitcode != 0 or "error" in item:
                    raise RuntimeError(item)
                outcome = np.asarray(item.pop("outcome"))
                if expected is None:
                    expected = outcome
                np.testing.assert_allclose(outcome, expected, rtol=1e-10, atol=1e-10, equal_nan=True)
                error = max(error, float(np.nanmax(np.abs(outcome-expected))))
                trials.append(item)
        stats = {backend: {key: summary([item[key] for item in trials if item["backend"] == backend])
                           for key in trials[0] if key != "backend"} for backend in args.backends}
        ratios = ({key: stats["pandas"][key]["median"]/stats["polars"][key]["median"]
                   for key in ("seconds", "explanation_seconds")} if {"pandas", "polars"}.issubset(stats) else {})
        report["families"][family] = {"trials": trials, "summary": stats, "pandas_time_over_polars_time": ratios,
                                       "max_absolute_parity_error": error}
        Path(args.output).write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        print(f"{family}: ratios {ratios}; max error {error}", flush=True)


if __name__ == "__main__":
    main()
