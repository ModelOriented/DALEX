"""Attribute backend overhead with cProfile; profiled times are not speed claims.

Run from the repository root. Uses identical seeded data, fitted estimator and
permutation settings for both backends. Writes dalex-backend-profile.json.
Use the isolated-process benchmark for end-to-end performance comparisons.
"""

import sys, cProfile, pstats, json, argparse, platform
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import dalex as dx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_variable_importance_backends import TimedPredict, weighted_rmse

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="dalex-backend-profile.json")
parser.add_argument("--revision", default="unspecified")
args = parser.parse_args()
rng = np.random.default_rng(123)
values = rng.normal(size=(10000, 10))
y = values[:, :5].sum(axis=1) + rng.normal(size=10000) * 0.1
names = [f"x{i}" for i in range(10)]
report = {
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source_revision": args.revision,
        "rows": 10000,
        "features": 10,
        "repetitions": 6,
        "rounds": 3,
        "seed": 42,
    }
}
with threadpool_limits(limits=1):
    model = HistGradientBoostingRegressor(max_iter=20, random_state=42).fit(values, y)
    for backend in ("pandas", "polars"):
        prof = cProfile.Profile()
        phases = []
        for repeat in range(6):
            predictor = TimedPredict()
            start = perf_counter()
            frame = (
                pd.DataFrame(values.copy(), columns=names)
                if backend == "pandas"
                else pl.DataFrame(values.copy(), schema=names)
            )
            constructed = perf_counter()
            prof.enable()
            explainer = dx.Explainer(
                model,
                frame,
                y,
                weights=np.linspace(1, 2, len(y)),
                predict_function=predictor,
                model_type="regression",
                verbose=False,
                precalculate=False,
            )
            initialized = perf_counter()
            result = explainer.model_parts(
                N=10000, B=3, random_state=42, loss_function=weighted_rmse
            )
            end = perf_counter()
            prof.disable()
            phases.append(
                dict(
                    construction=constructed - start,
                    initialization=initialized - constructed,
                    importance=end - initialized,
                    inference=predictor.seconds,
                )
            )
        stats = pstats.Stats(prof)
        top = []
        for (filename, line, function), (
            primitive,
            calls,
            own,
            cumulative,
            callers,
        ) in sorted(stats.stats.items(), key=lambda x: x[1][3], reverse=True)[:35]:
            top.append(
                dict(
                    file="/".join(Path(filename).parts[-3:]),
                    line=line,
                    function=function,
                    calls=calls,
                    own=own,
                    cumulative=cumulative,
                )
            )
        report[backend] = dict(phases=phases, hotspots=top)
Path(args.output).write_text(json.dumps(report, indent=2))
for backend in ("pandas", "polars"):
    record = report[backend]
    print(
        backend,
        {
            key: float(np.median([p[key] for p in record["phases"]]))
            for key in record["phases"][0]
        },
    )
    for hot in record["hotspots"][:22]:
        print(
            hot["function"],
            round(hot["own"], 4),
            round(hot["cumulative"], 4),
            hot["calls"],
        )
