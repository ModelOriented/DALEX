# Native Polars models with DALEX

These examples target this contribution branch. From the repository root, install
the Python package and the model libraries:

```sh
python -m pip install -e 'python/dalex[polars]' scikit-learn 'lightgbm>=4.7,<5'
```

## Numeric scikit-learn and LightGBM models

Fit the estimator on the same feature schema that its prediction method will
receive. DALEX keeps the feature batches in Polars; targets, losses and public
explanation tables use NumPy/pandas where their APIs require them.

```python
import numpy as np
import polars as pl
import dalex as dx
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor

rng = np.random.default_rng(123)
features = pl.DataFrame(rng.normal(size=(1_000, 3)), schema=["x", "z", "u"])
target = 2 * features["x"] - features["z"] + pl.Series(rng.normal(size=1_000) * .1)

models = [
    HistGradientBoostingRegressor(max_iter=20, random_state=42),
    LGBMRegressor(n_estimators=20, n_jobs=1, random_state=42, verbosity=-1),
]
for model in models:
    model.fit(features, target.to_numpy())
    explainer = dx.Explainer(model, features, target, verbose=False)
    importance = explainer.model_parts(N=500, B=3, random_state=42)
    attribution = explainer.predict_parts(
        features.head(1), type="shap", N=500, B=10, random_state=42,
    )
    profile = explainer.model_profile(
        N=100, variables=["x"], grid_points=11, random_state=42, verbose=False,
    )
    assert explainer.data is features
    print(type(model).__name__, importance.result)
```

For custom predictors, accept the native input batch and return a one-dimensional
prediction vector. A Polars Series or one-column frame is also normalized:

```python
def predict(model, batch):
    assert isinstance(batch, pl.DataFrame)
    return model.predict(batch)
```

An estimator's own internal NumPy/Arrow conversion still contributes to runtime.
Preserving a Polars input does not make inference itself faster. The committed
benchmarks separate dataframe work from inference and include memory results.

## sklearn preprocessing with Polars outputs

An sklearn `ColumnTransformer` can emit Polars batches for the final estimator.
DALEX still explains the original features, including one-hot encoded categories.
The tests exercise this route with sklearn and LightGBM regression/classification,
numeric imputation, unknown categories, profiles and Kernel SHAP. This example uses
the `features`, `target` and imports from the numeric example above:

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

preprocessing = ColumnTransformer([
    ("numeric", make_pipeline(SimpleImputer(), StandardScaler()), ["x", "z", "u"]),
]).set_output(transform="polars")
pipeline = make_pipeline(
    preprocessing,
    HistGradientBoostingRegressor(max_iter=20, random_state=42),
).fit(features, target.to_numpy())
explainer = dx.Explainer(pipeline, features, target, verbose=False)
importance = explainer.model_parts(N=500, B=3, random_state=42)
```

`set_output(transform="polars")` was introduced in sklearn 1.4, but Polars 2
removed the interchange/dimension APIs used to validate those outputs in older
sklearn. The direct recipe above is tested with sklearn 1.9. On 1.6–1.8, use
positional selectors and an explicit matrix boundary for preprocessing, followed
by a model-owned `FunctionTransformer` that creates a Polars batch before the
estimator. The integration tests exercise this fallback without changing DALEX's
native input batches. Arrow input selection in older sklearn versions has
separate limitations; the explicit positional preprocessing recipe below remains
available for that case.

## Categorical LightGBM features

LightGBM 4.7's native Arrow-stream path rejects raw dictionary/Enum columns. Use
an explicit, stable training-time encoding rather than asking DALEX to invent
category codes. Keep that mapping for future observations too.

```python
raw = pl.DataFrame({"group": ["a", "b", None, "a"] * 25,
                    "value": np.arange(100, dtype=float)})
features = raw.with_columns(
    pl.col("group").cast(pl.Enum(["a", "b"])).to_physical().cast(pl.Int32)
)
target = features["value"].to_numpy() + np.array([v == "b" for v in raw["group"]]) * 5
model = LGBMRegressor(n_estimators=20, n_jobs=1, random_state=42, verbosity=-1)
model.fit(features, target, categorical_feature=["group"])
explainer = dx.Explainer(model, features, target, verbose=False)
profile = explainer.model_profile(
    N=100, variables=["group"], variable_type="categorical",
    variable_splits={"group": [0, 1]}, random_state=42, verbose=False,
)
```

Integer codes still have numeric dataframe dtypes. Specify categorical grids
explicitly so profiles do not interpolate between codes. Likewise, provide
`categorical_features` to LIME when using encoded categorical inputs. Unknown
categories and missing values must follow the fitted model's encoding policy.

## Missing categories in sklearn pipelines

sklearn's OrdinalEncoder can learn Polars `None` as a category while treating
pandas categorical NaN as missing. Define the model's missing-category policy
explicitly when comparing backends. A native preprocessing step can standardize
the values before the estimator boundary:

```python
import narwhals.stable.v2 as nw

def normalize_group(frame):
    return nw.from_native(frame).with_columns(
        nw.when(nw.col("group").is_null()).then(nw.lit("__MISSING__"))
        .otherwise(nw.col("group").cast(nw.String)).alias("group")
    ).to_native()
```

Choose a sentinel outside the real category domain. Older sklearn
ColumnTransformer versions cannot index Arrow Tables; an explicit NumPy boundary
inside the fitted preprocessing pipeline is covered by the integration tests.

Lazy inputs require the caller to collect explicitly. Distributed/GPU backends
and arbitrary model-specific dtype policies have separate requirements; the
validation report states the combinations actually tested.
