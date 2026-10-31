"""Reproduce the regression fixture before any DALEX import or explanation."""
import json
import sys
from importlib.metadata import version
import numpy as np
import pandas as pd
import polars as pl
import narwhals.stable.v2 as nw
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMRegressor
from threadpoolctl import threadpool_limits

def polars_features(frame):
    return frame if isinstance(frame, pl.DataFrame) else pl.from_arrow(frame)

def years(frame):
    return nw.from_native(frame).select(nw.col('when').dt.year()).to_numpy()

def transformed_polars(values):
    frame = pl.DataFrame(values, schema=['x', 'z', 'kind_a', 'kind_b', 'when'], orient='row')
    print('transformed', values.shape, values.dtype, values.flags.c_contiguous, values.flags.f_contiguous,
          frame.shape, frame.schema, np.min(values, axis=0), np.max(values, axis=0), flush=True)
    return frame

print(json.dumps({name: version(name) for name in ['numpy','pandas','polars','pyarrow','narwhals','scikit-learn','lightgbm']}), flush=True)
rng = np.random.default_rng(615)
data = pl.DataFrame({'x': rng.normal(size=64), 'z': rng.normal(size=64),
    'kind': ['a', 'b'] * 32,
    'when': pd.date_range('2020-01-01', periods=64, freq='ns', tz='UTC'),
    'group': pl.Series(['u', None, 'u', None] * 16).cast(pl.Enum(['u','unused']))})
target = 2 * data['x'].to_numpy() - data['z'].to_numpy() + np.tile([0., 2.], 32)
preparation = ColumnTransformer([
    ('numeric', make_pipeline(SimpleImputer(), StandardScaler()), [0,1]),
    ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [2]),
    ('time', FunctionTransformer(years, validate=False), [3]),
])
with threadpool_limits(limits=1):
    model = make_pipeline(FunctionTransformer(polars_features, validate=False), preparation,
        FunctionTransformer(transformed_polars, validate=False),
        LGBMRegressor(n_estimators=8,n_jobs=1,random_state=42,verbosity=-1)).fit(data,target)
    assert np.isfinite(model.predict(data)).all()
assert 'dalex' not in sys.modules
print('Standalone fitted pipeline passed', flush=True)
