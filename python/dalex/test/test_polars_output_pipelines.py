"""Explain original features while sklearn preprocessing emits native Polars."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

import dalex as dx


@pytest.mark.parametrize('library', ['sklearn', 'lightgbm'])
@pytest.mark.parametrize('classification', [False, True])
@pytest.mark.parametrize('family', ['importance', 'break_down', 'shap', 'partial',
                                    'accumulated', 'conditional', 'categorical_profile', 'kernel_shap'])
def test_polars_output_pipeline_explains_original_features(library, classification, family):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
    import sklearn
    from packaging.version import Version
    from threadpoolctl import threadpool_limits

    rng = np.random.default_rng(215)
    x = rng.normal(size=80)
    x[::9] = np.nan
    data = pl.DataFrame({'x': x, 'z': rng.normal(size=80), 'kind': ['a', 'b'] * 40}).with_columns(
        pl.col('kind').cast(pl.Enum(['a', 'b', 'unseen'])))
    response = np.nan_to_num(x) * 2 - data['z'].to_numpy() + np.tile([0., 2.], 40)
    y = (response > np.median(response)).astype(int) if classification else response
    preprocessing = ColumnTransformer([
        # Older sklearn selects Polars through its former interchange protocol;
        # positional selectors also work with Polars 2, which removed that API.
        ('numeric', make_pipeline(SimpleImputer(), StandardScaler()), [0, 1]),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [2]),
    ], verbose_feature_names_out=False)
    legacy_polars = Version(sklearn.__version__) < Version('1.9')
    if not legacy_polars:
        preprocessing.set_output(transform='polars')
    if library == 'lightgbm':
        from lightgbm import LGBMClassifier, LGBMRegressor
        estimator = (LGBMClassifier if classification else LGBMRegressor)(
            n_estimators=8, n_jobs=1, verbosity=-1, random_state=42)
    else:
        estimator = (HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor)(
            max_iter=8, random_state=42)
    with threadpool_limits(limits=1):
        if legacy_polars:
            # sklearn <1.9 validates dataframe outputs using the interchange
            # protocol removed by Polars 2. Keep that external matrix boundary
            # explicit, then give the estimator a model-owned Polars batch.
            bridge = FunctionTransformer(lambda values: pl.DataFrame(
                values, schema=['x', 'z', 'kind_a', 'kind_b'], orient='row'))
            model = make_pipeline(preprocessing, bridge, estimator).fit(data, y)
        else:
            model = make_pipeline(preprocessing, estimator).fit(data, y)
        method = 'predict_proba' if classification else 'predict'
        original_predict = getattr(estimator, method)
        seen = []
        def transformed_predict(batch, *args, **kwargs):
            assert isinstance(batch, pl.DataFrame)
            assert list(batch.columns) == ['x', 'z', 'kind_a', 'kind_b']
            seen.append(len(batch))
            return original_predict(batch, *args, **kwargs)
        setattr(estimator, method, transformed_predict)
        results = []
        # An unused training-schema Enum level is a valid, unknown one-hot level.
        observation = data.head(1).with_columns(pl.lit('unseen').cast(data.schema['kind']).alias('kind'))
        for frame, obs in ((data.to_pandas(), observation.to_pandas()), (data, observation)):
            def predict(model, batch):
                assert type(batch) is type(frame)
                return model.predict_proba(batch)[:, 1] if classification else model.predict(batch)
            exp = dx.Explainer(model, frame, y, predict_function=predict, label='pipeline', verbose=False)
            assert exp.data is frame
            if family == 'importance':
                result = exp.model_parts(N=30, B=2, random_state=42).result
            elif family in ('break_down', 'shap'):
                result = exp.predict_parts(obs, type=family, N=30, B=3, random_state=42).result
            elif family == 'kernel_shap':
                explanation = exp.predict_parts(obs, type='shap_wrapper', shap_explainer_type='KernelExplainer',
                                                N=20, random_state=42, nsamples=20, l1_reg=0, silent=True)
                values = np.asarray(explanation.result)
                np.testing.assert_allclose(values.sum() + np.asarray(explanation.expected_value).item(),
                                           exp.predict(obs)[0], atol=1e-10, rtol=1e-10)
                results.append(values)
                continue
            elif family == 'categorical_profile':
                result = exp.model_profile(type='partial', N=20, variables=['kind'], variable_type='categorical',
                                           variable_splits={'kind': ['a', 'b', 'unseen']},
                                           random_state=42, verbose=False).result
                assert set(result['_x_']) == {'a', 'b', 'unseen'}
            else:
                result = exp.model_profile(type=family, N=20, variables=['z'], grid_points=4,
                                           random_state=42, verbose=False).result
            results.append(result.select_dtypes(include=np.number).to_numpy(dtype=float))
        assert seen
        np.testing.assert_allclose(results[0], results[1], atol=1e-10, rtol=1e-10, equal_nan=True)
        assert data['kind'].to_list() == ['a', 'b'] * 40
