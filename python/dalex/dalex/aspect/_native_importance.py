"""Native feature interventions and vector reductions for Aspect importance."""
import multiprocessing as mp
import numpy as np
import narwhals.stable.v2 as nw
from .._dataframe import take_rows, set_columns, column_values, observation_values, copy_for_edit
from .._random import get_random_state
from ._predict_aspect_importance.utils import get_sample


def descending(values):
    values = np.asarray(values)
    valid = np.flatnonzero(~np.isnan(values))
    order = len(valid) - 1 - np.argsort(np.abs(values[valid])[::-1], kind='quicksort')[::-1]
    return np.r_[valid[order], np.flatnonzero(np.isnan(values))]


def path_contributions(predict, sample, cells, groups, order, *, reset_index=False):
    """Apply an explicit group path and return its ordered numeric changes."""
    data = copy_for_edit(sample)
    if reset_index:
        from narwhals.dependencies import is_pandas_dataframe
        if is_pandas_dataframe(data):
            data = data.reset_index(drop=True)
    predictions = [predict(data).mean()]
    for i in order:
        for name in groups[i]:
            data = set_columns(data, {name: cells[name]}, scalar=True, copy=False)
        predictions.append(predict(data).mean())
    return np.diff(predictions)


def path(predict, sample, cells, groups, rng):
    order = rng.choice(len(groups), len(groups), replace=False)
    values = np.empty(len(groups))
    values[order] = path_contributions(predict, sample, cells, groups, order, reset_index=True)
    return values


def calculate(explainer, observation, groups, *, kind, N, n_aspects=None,
              sample_method='default', f=2, B=25, processes=1, random_state=None, rng=None):
    cells = observation_values(observation)
    names = list(groups)
    variables = list(groups.values())
    r = np.random.RandomState(random_state)
    count = min(len(explainer.data), N)
    sample = take_rows(explainer.data, r.randint(0, len(explainer.data), count))
    if kind == 'default' and len(groups) == 1:
        baseline = (explainer.y_hat if explainer.y_hat is not None else explainer.predict(explainer.data)).mean()
        coefficients = explainer.predict(observation) - baseline
    elif kind == 'default':
        matrix = get_sample(count, len(groups), sample_method, f, random_state)
        replacements = {}
        for i, group in enumerate(variables):
            for name in group:
                values = replacements.get(name, column_values(sample, name, preserve_nulls=True))
                replacements[name] = np.where(matrix[:, i] == 1, cells[name], values)
        changed = set_columns(copy_for_edit(sample), replacements, copy=False)
        from narwhals.dependencies import is_pandas_dataframe
        if is_pandas_dataframe(changed):
            # The original local surrogate reset its intervention index while
            # retaining sampled labels on the reference batch. Index-sensitive
            # pandas callbacks rely on that existing public contract.
            changed = changed.reset_index(drop=True)
        response = explainer.predict(changed) - explainer.predict(sample)
        if n_aspects is None:
            from sklearn.linear_model import LinearRegression
            coefficients = LinearRegression().fit(matrix, response).coef_
        else:
            from sklearn.linear_model import lasso_path
            estimates = lasso_path(matrix, response)[1]
            index = np.max(np.where(np.count_nonzero(estimates, axis=0) <= n_aspects))
            coefficients = estimates[:, index]
    else:
        if B < 1:
            raise ValueError('B must be positive')
        if processes == 1:
            rng = get_random_state(random_state) if rng is None else rng
            draws = [path(explainer.predict, sample, cells, variables, rng) for _ in range(B)]
        else:
            generators = [np.random.default_rng(seed) for seed in np.random.SeedSequence(random_state).spawn(B)]
            with mp.get_context('spawn').Pool(processes) as pool:
                draws = pool.starmap(path, [(explainer.predict, sample, cells, variables, rng) for rng in generators])
        coefficients = np.nanmean(np.asarray(draws), axis=0)
        # pandas' pivot sorted aspect names before the legacy descending sort.
        sorted_names = sorted(range(len(names)), key=lambda i: names[i])
        names = [names[i] for i in sorted_names]
        variables = [variables[i] for i in sorted_names]
        coefficients = coefficients[sorted_names]
    order = descending(coefficients)
    backend = nw.get_native_namespace(nw.from_native(explainer.data, eager_only=True))
    data = nw.from_dict({'aspect_name': [names[i] for i in order],
                         'importance': np.asarray(coefficients)[order],
                         'label': [explainer.label] * len(order)}, backend=backend).to_native()
    return data, [variables[i] for i in order], kind == 'default' and len(groups) == 1
