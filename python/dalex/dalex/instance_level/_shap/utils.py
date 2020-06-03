import multiprocessing as mp
from copy import deepcopy

import numpy as np
import pandas as pd


def iterate_paths(predict_function, model, data, label, new_observation, p, b):
    random_path = np.random.choice(np.arange(p), p, replace=False)
    return get_single_random_path(predict_function, model, data, label, new_observation, random_path, b)


def shap(explainer,
         new_observation,
         path,
         keep_distributions,
         B,
         processes):
    # Now we know the path, so we can calculate contributions
    # set variable indicators
    # start random path
    p = new_observation.shape[1]

    if processes == 1:
        result = [
            iterate_paths(explainer.predict_function, explainer.model, explainer.data, explainer.label, new_observation,
                          p, b + 1)
            for b in range(B)]
    else:
        pool = mp.Pool(processes)
        result = pool.starmap_async(iterate_paths,
                                    [(explainer.predict_function, explainer.model, explainer.data, explainer.label,
                                      new_observation, p, b + 1) for b in range(B)]).get()
        pool.close()

    if path is not None:
        if isinstance(path, str) and path == 'average':
            # let's calculate an average attribution
            extracted_contributions = [None] * B
            for i in range(B):
                extracted_contributions[i] = result[i].sort_values(['label', 'variable']).loc[:, 'contribution']

            extracted_contributions = pd.concat(extracted_contributions, axis=1)
            result_average = deepcopy(result[0])
            result_average = result_average.sort_values(['label', 'variable'])
            result_average['contribution'] = extracted_contributions.mean(axis=1)
            result_average['B'] = 0
            result_average['sign'] = np.sign(result_average['contribution'])

            result.append(result_average)
        else:
            tmp = get_single_random_path(explainer.predict_function, explainer.model, explainer.data, explainer.label,
                                         new_observation, path, 0)
            # tmp['B'] = 0

            result.append(tmp)

    result = pd.concat(result)
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
    return str(x) if isinstance(x, (str, np.str_)) else str(float(signif(x)))


#:# https://stackoverflow.com/a/59888924
def signif(x, p=4):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags
