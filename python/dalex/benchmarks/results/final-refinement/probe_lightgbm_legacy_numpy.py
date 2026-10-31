"""Standalone dependency probe: deliberately does not import DALEX/Narwhals."""
import argparse
import json
import sys
from contextlib import nullcontext
from importlib.metadata import version
import numpy as np
from lightgbm import LGBMRegressor
from threadpoolctl import threadpool_limits

parser = argparse.ArgumentParser()
parser.add_argument('--threadpoolctl', action='store_true')
parser.add_argument('--backend', choices=['numpy', 'polars', 'pyarrow'], default='numpy')
args = parser.parse_args()
print(json.dumps({'python': sys.version, 'numpy': np.__version__, 'numpy_path': np.__file__,
    'lightgbm': version('lightgbm'), 'scikit-learn': version('scikit-learn'),
    'scipy': version('scipy'), 'polars': version('polars'), 'pyarrow': version('pyarrow'),
    'narwhals': version('narwhals'), 'backend': args.backend, 'threadpoolctl': args.threadpoolctl}), flush=True)
rng = np.random.default_rng(615)
data = rng.normal(size=(64, 5))
target = data[:, 0] * 2 - data[:, 1]
if args.backend != 'numpy':
    import polars as pl
    data = pl.DataFrame(data, schema=[f'x{i}' for i in range(5)])
    if args.backend == 'pyarrow':
        data = data.to_arrow()
with threadpool_limits(limits=1) if args.threadpoolctl else nullcontext():
    model = LGBMRegressor(n_estimators=8, n_jobs=1, random_state=42, verbosity=-1).fit(data, target)
    assert np.isfinite(model.predict(data)).all()
assert 'dalex' not in sys.modules
print('Standalone LightGBM fit/predict passed', flush=True)
