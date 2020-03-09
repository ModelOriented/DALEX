import pandas as pd
import numpy as np
from copy import deepcopy


def shap(explainer,
         new_observation,
         path,
         keep_distributions,
         B):

    # Now we know the path, so we can calculate contributions
    # set variable indicators
    # start random path
    p = new_observation.shape[1]

    result = [None] * B
    for b in range(B):
        random_path = np.random.choice(np.arange(p), p, replace=False)
        tmp = get_single_random_path(explainer, new_observation, random_path)
        tmp['B'] = b
        result[b] = tmp

    if path is not None:
        if path == 'average':
            # let's calculate an average attribution
            extracted_contributions = [None] * B
            for i in range(B):
                extracted_contributions[i] = result[i].sort_values(['label', 'variable']).loc[:, 'contribution']

            extracted_contributions = pd.concat(extracted_contributions, axis=1)
            result_average = deepcopy(result[0])
            result_average = result_average.sort_values(['label', 'variable'])
            result_average['contribution'] = extracted_contributions.mean(axis=1)
            result_average['B'] = -1

            result = result.append(result_average)
        else:
            tmp = get_single_random_path(explainer, new_observation, path)
            tmp['B'] = -1
            result = result.append(tmp)

    result = pd.concat(result)
    if keep_distributions:
        yhats_distributions = calculate_yhats_distributions(explainer)
    else:
        yhats_distributions = None

    target_yhat = explainer.predict(new_observation)
    yhatpred = explainer.predict(explainer.data)
    baseline_yhat = yhatpred.mean()

    return result, target_yhat, baseline_yhat, yhats_distributions


def calculate_yhats_distributions(explainer):
    pred = explainer.predict(explainer.data)

    return pd.DataFrame({
        'variable_name': 'all_data',
        'variable': 'all data',
        'id': np.arange(explainer.data.shape[0]),
        'prediction': pred,
        'label': explainer.label
    })


def get_single_random_path(explainer, new_observation, random_path):
    current_data = deepcopy(explainer.data)
    yhats = np.empty(len(random_path) + 1)
    yhats[0] = explainer.predict(current_data).mean()
    for i, candidate in enumerate(random_path):
        current_data.iloc[:, candidate] = new_observation.iloc[0, candidate]
        yhats[i+1] = explainer.predict(current_data).mean()

    diffs = np.diff(yhats)

    variable_names = explainer.data.columns[random_path]
    
    new_observation_f = new_observation.loc[:, variable_names]\
        .apply(lambda x: nice_format(x.iloc[0]))

    return pd.DataFrame({
        'variable': [' = '.join(pair) for pair in zip(variable_names, new_observation_f)],
        'contribution': diffs,
        'variable_name': variable_names,
        'variable_value': new_observation.loc[:, variable_names].values.reshape(-1,),
        'sign': np.sign(diffs),
        'label': explainer.label
    })


def nice_pair(df, i1, i2):
    return nice_format(df.iloc[0, i1]) if i2 is None else ':'.join(
        (nice_format(df.iloc[0, i1]), nice_format(df.iloc[0, i2])))


def nice_format(x):
    return str(x) if isinstance(x, (str, np.str_)) else str.format('{0:.2}', float(x))
