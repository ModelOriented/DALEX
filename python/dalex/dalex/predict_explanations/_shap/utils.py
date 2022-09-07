import multiprocessing as mp
from copy import deepcopy
from numpy.random import SeedSequence, default_rng

import numpy as np
import pandas as pd


def iterate_paths(predict_function, model, data, label, new_observation, p, b, rng):
    random_path = rng.choice(np.arange(p), p, replace=False)
    return get_single_random_path(predict_function, model, data, label, new_observation, random_path, b)


def shap(explainer,
         new_observation,
         path,
         keep_distributions,
         B,
         processes,
         random_state):
    # Now we know the path, so we can calculate contributions
    # set variable indicators
    # start random path
    p = new_observation.shape[1]

    if processes == 1:
        result_list = [
            iterate_paths(explainer.predict_function, explainer.model, explainer.data,
                          explainer.label, new_observation, p, b + 1, np.random)
            for b in range(B)]
    else:
        # Create number generator for each iteration
        ss = SeedSequence(random_state)
        generators = [default_rng(s) for s in ss.spawn(B)]
        pool = mp.get_context('spawn').Pool(processes)
        result_list = pool.starmap_async(iterate_paths,
                                         [(explainer.predict_function, explainer.model, explainer.data,
                                           explainer.label, new_observation, p, b + 1, generators[b]) for b in range(B)]).get()
        pool.close()

    result = pd.concat(result_list)

    if path is not None:
        if isinstance(path, str) and path == 'average':
            # average over all of the paths
            variable_average = result.pivot(index='variable', columns='B', values='contribution').mean(axis=1)
            # sort pd.Series by index of abs-sorted pd.Series
            variable_average_sorted = \
                variable_average.reindex(variable_average.abs().sort_values(ascending=False).index)
            # make the final result - sort and fill with values
            result_average = result_list[0].set_index('variable').reindex(variable_average_sorted.index).reset_index()

            result_average = result_average.assign(contribution=variable_average_sorted.values,
                                                   B=0,
                                                   sign=np.sign(variable_average_sorted.values))

            result = pd.concat((result, result_average))
        else:
            tmp = get_single_random_path(explainer.predict_function, explainer.model, explainer.data, explainer.label,
                                         new_observation, path, 0)

            result = pd.concat((result, tmp))

    if keep_distributions:
        yhats_distributions = calculate_yhats_distributions(explainer)
    else:
        yhats_distributions = None

    target_yhat = explainer.predict(new_observation)[0]  # only one new_observation allowed
    data_yhat = explainer.predict(explainer.data)
    baseline_yhat = data_yhat.mean()

    return result, target_yhat, baseline_yhat, yhats_distributions


def calculate_yhats_distributions(explainer):
    data_yhat = explainer.predict(explainer.data)

    return pd.DataFrame({
        'variable_name': 'all_data',
        'variable': 'all data',
        'id': np.arange(explainer.data.shape[0]),
        'prediction': data_yhat,
        'label': explainer.label
    })


def get_single_random_path(predict, model, data, label, new_observation, random_path, b):
    current_data = deepcopy(data)
    yhats = np.empty(len(random_path) + 1)
    yhats[0] = predict(model, current_data).mean()
    for i, candidate in enumerate(random_path):
        current_data.iloc[:, candidate] = new_observation.iloc[0, candidate]
        yhats[i + 1] = predict(model, current_data).mean()

    diffs = np.diff(yhats)

    variable_names = data.columns[random_path]

    new_observation_f = new_observation.loc[:, variable_names] \
        .apply(lambda x: nice_format(x.iloc[0]))

    return pd.DataFrame({
        'variable': [' = '.join(pair) for pair in zip(variable_names, new_observation_f)],
        'contribution': diffs,
        'variable_name': variable_names,
        'variable_value': new_observation.loc[:, variable_names].values.reshape(-1, ),
        'sign': np.sign(diffs),
        'label': label,
        'B': b
    })


def nice_pair(df, i1, i2):
    return nice_format(df.iloc[0, i1]) if i2 is None else ':'.join(
        (nice_format(df.iloc[0, i1]), nice_format(df.iloc[0, i2])))


def nice_format(x):
    return str(x) if isinstance(x, str) else str(float(signif(x)))


#:# https://stackoverflow.com/a/59888924
def signif(x, p=4):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags
