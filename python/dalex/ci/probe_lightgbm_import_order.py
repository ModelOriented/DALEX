"""Isolate a Windows legacy dependency failure without importing DALEX.

Run this with the interpreter/environment under investigation. Every case gets a
fresh subprocess, so a failed native training call cannot contaminate another.
The report is written even on failure; a nonzero exit means at least one case
failed. This is a dependency diagnostic, not a waived DALEX test.
"""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def child(backend, first):
    if first == "pandas":
        import pandas as pd
        from lightgbm import LGBMRegressor
    else:
        from lightgbm import LGBMRegressor
        import pandas as pd
    import numpy as np

    assert "dalex" not in sys.modules
    values = np.arange(160, dtype=float).reshape(80, 2)
    target = np.arange(80, dtype=float)
    data = values
    if backend == "pandas":
        data = pd.DataFrame(values, columns=["x", "z"])
    elif backend == "polars":
        import polars as pl
        data = pl.DataFrame(values, schema=["x", "z"])
    model = LGBMRegressor(n_estimators=2, n_jobs=1, verbosity=-1, random_state=42)
    model.fit(data, target)
    return {"prediction": model.predict(data).tolist(), "dalex_imported": "dalex" in sys.modules}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="lightgbm-import-order.json")
    parser.add_argument("--child-backend", choices=["numpy", "pandas", "polars"])
    parser.add_argument("--first", choices=["pandas", "lightgbm"])
    args = parser.parse_args()
    if args.child_backend:
        try:
            print(json.dumps({"passed": True, **child(args.child_backend, args.first)}))
        except Exception as exc:
            print(json.dumps({"passed": False, "error": f"{type(exc).__name__}: {exc}"}))
            sys.exit(1)
        return

    versions = {}
    for name in ("pandas", "numpy", "polars", "pyarrow", "narwhals", "scipy", "scikit-learn", "lightgbm"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    report = {"python": sys.version, "platform": platform.platform(), "versions": versions,
              "description": "Standalone dependency training; no DALEX import in child processes", "cases": []}
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for first in ("pandas", "lightgbm"):
        for backend in ("numpy", "pandas", "polars"):
            result = subprocess.run(
                [sys.executable, __file__, "--child-backend", backend, "--first", first],
                env=env, capture_output=True, text=True, timeout=60,
            )
            item = {"first_import": first, "backend": backend, "returncode": result.returncode}
            try:
                item.update(json.loads(result.stdout.splitlines()[-1]))
            except (ValueError, IndexError):
                item.update(passed=False, error="child did not return JSON")
            if result.stderr:
                item["stderr_tail"] = result.stderr.splitlines()[-10:]
            report["cases"].append(item)
    report["passed"] = all(item.get("passed") and item["returncode"] == 0 for item in report["cases"])
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "cases": [
        {key: item.get(key) for key in ("first_import", "backend", "passed", "error")}
        for item in report["cases"]]}, indent=2))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
