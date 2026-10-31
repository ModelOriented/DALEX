"""Record automatic GIL fallback separately from explicit PYTHON_GIL=0 tests.

Each import gets its own interpreter so one dependency's fallback cannot hide
another's. This is dependency evidence, not a claim of concurrent thread safety.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="gil-import-probe.json")
    args = parser.parse_args()
    records = []
    environment = dict(os.environ)
    environment.pop("PYTHON_GIL", None)
    for name in (
        "numpy", "scipy", "pandas", "polars", "narwhals", "pyarrow", "sklearn", "lightgbm",
        "shap", "lime.lime_tabular", "skimage", "statsmodels.api", "matplotlib",
        "flask", "flask_cors", "requests", "IPython", "ipywidgets", "anywidget", "kaleido", "dalex",
    ):
        code = (
            "import importlib,json,sys,sysconfig; "
            "before=sys._is_gil_enabled(); "
            f"importlib.import_module({name!r}); "
            "print(json.dumps(dict(free_threaded=bool(sysconfig.get_config_var('Py_GIL_DISABLED')),"
            "gil_before=before,gil_after=sys._is_gil_enabled())))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], env=environment,
            capture_output=True, text=True, timeout=120,
        )
        record = {"module": name, "returncode": result.returncode, "stderr": result.stderr}
        if result.returncode == 0:
            record.update(json.loads(result.stdout.splitlines()[-1]))
            assert record["free_threaded"] and not record["gil_before"], record
        else:
            record["stdout"] = result.stdout
        records.append(record)
        print(json.dumps(record), flush=True)
    Path(args.output).write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    assert all(record["returncode"] == 0 for record in records), "A dependency failed to import"


if __name__ == "__main__":
    main()
