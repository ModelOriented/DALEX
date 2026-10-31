"""Legacy pandas-returning adapters over shared contribution kernels."""
import numpy as np
import narwhals.stable.v2 as nw
from ..._dataframe import (as_eager_dataframe, column_names, observation_values,
                           observation_display_values, to_pandas)


def iterate_paths(predict_function, model, data, label, new_observation, p, b, rng, values=None):
    random_path = rng.choice(p, p, replace=False)
    return get_single_random_path(predict_function, model, data, label,
                                   new_observation, random_path, b, values)


def shap(explainer, new_observation, path, keep_distributions, B, processes, random_state):
    from .._native_contributions import shap as native_shap
    result, target, baseline, distribution = native_shap(explainer, new_observation,
        path, keep_distributions, B, processes, random_state)
    return (result._dalex_pandas_view(), target, baseline,
            to_pandas(distribution) if distribution is not None else None)


def calculate_yhats_distributions(explainer):
    from .._native_contributions import _distribution
    namespace = nw.get_native_namespace(as_eager_dataframe(explainer.data))
    return to_pandas(_distribution('all_data', 'all data', explainer.predict(explainer.data),
                                   explainer.label, namespace))


def get_single_random_path(predict, model, data, label, new_observation, random_path, b, values=None):
    from .._native_contributions import _shap_path, ShapTable, _format
    cells = observation_values(new_observation) if values is None else values
    display = observation_display_values(new_observation, cells)
    names = list(column_names(data))
    variables = [names[int(i)] for i in random_path]
    contributions = _shap_path(predict, model, data, names, cells, random_path)
    namespace = nw.get_native_namespace(as_eager_dataframe(data))
    result = nw.from_dict({'variable': [str(name) + ' = ' + _format(display[name]) for name in variables],
        'contribution': contributions, 'variable_name': variables, 'sign': np.sign(contributions),
        'label': [label] * len(variables), 'B': [b] * len(variables)}, backend=namespace).to_native()
    return ShapTable(result, new_observation)._dalex_pandas_view()


def nice_pair(df, i1, i2):
    return nice_format(df.iloc[0, i1]) if i2 is None else ':'.join(
        (nice_format(df.iloc[0, i1]), nice_format(df.iloc[0, i2])))


def nice_format(x):
    return str(float(signif(x))) if isinstance(x, (int, float, np.integer, np.floating)) else str(x)


#:# https://stackoverflow.com/a/59888924
def signif(x, p=4):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags
