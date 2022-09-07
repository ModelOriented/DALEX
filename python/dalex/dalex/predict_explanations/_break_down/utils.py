from copy import deepcopy

import numpy as np
import pandas as pd


def local_interactions(explainer,
                       new_observation,
                       interaction_preference,
                       type,
                       order,
                       keep_distributions):
    # here one can add model and data and new observation
    # just in case only some variables are specified
    # this will work only for data.frames

    # set target
    target_yhat = explainer.predict(new_observation)
    baseline_yhat = np.mean(explainer.predict(explainer.data))

    # 1d changes
    # how the average would change if single variable is changed
    average_yhats = calculate_1d_changes(explainer, new_observation)
    diffs_1d = average_yhats - baseline_yhat

    if type == '2d':
        # impact summary for 1d variables
        feature_path_1d = pd.DataFrame({
            'diff': diffs_1d,
            'adiff': np.abs(diffs_1d),
            'diff_norm': diffs_1d,
            'adiff_norm': np.abs(diffs_1d),
            'ind1': np.arange(len(explainer.data.columns)),
            'ind2': None
        })

        p = explainer.data.shape[1]
        inds = pd.DataFrame({
            'ind1': np.hstack([np.arange(i, p) for i in range(1, p)]),
            'ind2': np.hstack([np.repeat(i - 1, p - i) for i in range(1, p)])
        })

        # 2d changes
        # how the average would change if two variables are changed
        average_yhats_2d, average_yhats_2d_norm = calculate_2d_changes(explainer, new_observation, inds, diffs_1d)

        diffs_2d = average_yhats_2d - baseline_yhat
        diffs_2d_norm = average_yhats_2d_norm - baseline_yhat

        # impact summary for 2d variables
        # large interaction_preference force to use interactions

        feature_path_2d = pd.DataFrame({
            'diff': diffs_2d,
            'adiff': np.abs(diffs_2d) * interaction_preference,
            'diff_norm': diffs_2d_norm,
            'adiff_norm': np.abs(diffs_2d_norm) * interaction_preference,
            'ind1': inds.loc[:, 'ind1'].values,
            'ind2': inds.loc[:, 'ind2'].values
        })

        feature_path = pd.concat((feature_path_1d, feature_path_2d))

    else:
        # different definition
        feature_path = pd.DataFrame({'diff': diffs_1d, 'ind1': np.arange(diffs_1d.shape[0])})

    # how variables shall be ordered in the BD plot?
    feature_path = create_ordered_path(feature_path, average_yhats.index, type, order)

    # Now we know the path, so we can calculate contributions
    # set variable indicators
    result, yhats = calculate_contributions_along_path(explainer,
                                                       new_observation,
                                                       feature_path,
                                                       baseline_yhat,
                                                       target_yhat,
                                                       type,
                                                       keep_distributions)

    result = pd.DataFrame(result)

    if keep_distributions:
        yhats_distributions = pd.DataFrame({
            'variable_name': "all data",
            'variable': "all data",
            'id': np.arange(0, explainer.data.shape[0]),
            'prediction': explainer.predict(explainer.data),
            'label': explainer.label
        })

        yhats_distributions = pd.concat([yhats_distributions, yhats])

    yhats_distributions = yhats_distributions if keep_distributions else None

    return result, yhats_distributions


def calculate_1d_changes(explainer,
                         new_observation):
    yhats = np.empty_like(explainer.data.columns)
    for i, col in enumerate(explainer.data.columns):
        current_data = deepcopy(explainer.data)
        current_data.loc[:, col] = new_observation.loc[:, col].iloc[0]
        yhats[i] = explainer.predict(current_data).mean()

    return pd.Series(yhats, index=explainer.data.columns)


def calculate_2d_changes(explainer,
                         new_observation,
                         inds,
                         diffs_1d):
    average_yhats = np.empty(inds.shape[0])
    average_yhats_norm = np.empty(inds.shape[0])

    for i in range(inds.shape[0]):
        current_data = deepcopy(explainer.data)
        current_data.iloc[:, inds.iloc[i, 0]] = new_observation.iloc[0, inds.iloc[i, 0]]
        current_data.iloc[:, inds.iloc[i, 1]] = new_observation.iloc[0, inds.iloc[i, 1]]

        yhats = explainer.predict(current_data)
        average_yhats[i] = yhats.mean()
        average_yhats_norm[i] = average_yhats[i] - diffs_1d[inds.iloc[i, 0]] - diffs_1d.iloc[inds.iloc[i, 1]]

    columns = explainer.data.columns
    average_yhats = pd.Series(average_yhats)
    average_yhats_norm = pd.Series(average_yhats_norm)
    average_yhats.index = [':'.join(pair) for pair in zip(columns[inds.iloc[:, 0]], columns[inds.iloc[:, 1]])]
    average_yhats_norm.index = [':'.join(pair) for pair in zip(columns[inds.iloc[:, 0]], columns[inds.iloc[:, 1]])]

    return average_yhats, average_yhats_norm


def create_ordered_path(feature_path,
                        average_yhats_index,
                        type,
                        order):
    if order is None:
        # sort impacts and look for most importants elements
        if type == '2d':
            feature_path = feature_path.iloc[np.argsort(feature_path['adiff_norm'])[::-1], :]
        else:
            feature_path = feature_path.iloc[np.argsort(feature_path['diff'])[::-1], :]
    elif np.issubdtype(order.dtype, int):
        # case when permutation
        feature_path = feature_path.iloc[order, :]

    elif np.isin(order, average_yhats_index).all():
        # case when character
        feature_path = feature_path.loc[order, :]
    else:
        raise ValueError('Wrong order!')

    return feature_path


def calculate_contributions_along_path(explainer,
                                       new_observation,
                                       feature_path,
                                       baseline_yhat,
                                       target_yhat,
                                       type,
                                       keep_distributions):
    open_variables = set(np.arange(explainer.data.shape[1]))
    current_data = deepcopy(explainer.data)
    step = 0
    yhats = []
    yhats_mean = []
    selected_rows = []

    for i in range(feature_path.shape[0]):
        candidates = [feature_path['ind1'].iloc[i]]
        if type == '2d':
            ind2_is_None = feature_path['ind2'].iloc[i] is None
            candidates.append(feature_path['ind2'].iloc[i]) if not ind2_is_None else None
        else:
            ind2_is_None = True

        if open_variables.issuperset(candidates):
            # we can add this effect to out path
            for candidate in candidates:
                current_data.iloc[:, candidate] = new_observation.iloc[0, candidate]

            step += 1

            yhats_pred = explainer.predict(current_data)

            if keep_distributions:
                yhats.append(
                    pd.DataFrame({
                        'variable_name': ':'.join(explainer.data.columns[candidates]),
                        'variable': ':'.join(explainer.data.columns[candidates]) +
                                    '=' +
                                    nice_pair(new_observation,
                                              candidates[0],
                                              None if ind2_is_None else candidates[1]),
                        'id': np.arange(explainer.data.shape[0]),
                        'prediction': yhats_pred,
                        'label': explainer.label
                    })
                )

            yhats_mean.append(np.mean(yhats_pred))
            selected_rows.append(i)
            open_variables = open_variables.difference(candidates)

    selected = feature_path.iloc[selected_rows, :]

    # extract values
    selected_values = selected.apply(lambda row: nice_pair(new_observation,
                                                           row['ind1'],
                                                           None if type == '1d' else row['ind2']),
                                     axis=1)

    # prepare values
    yhats = pd.concat(yhats) if keep_distributions else None
    
    cumulative = np.hstack((baseline_yhat, yhats_mean, target_yhat))
    contribution = np.hstack((cumulative[0], np.diff(cumulative)))
    contribution[-1] = cumulative[-1]
    
    result = {
        'variable_name': np.hstack(("intercept", selected.index, "")),
        'variable_value': np.hstack(("", selected_values, "")),
        'variable': 
            ["intercept"] + 
            [' = '.join(pair) for pair in zip(selected.index, selected_values)] + 
            ['prediction'],
        'cumulative': cumulative,
        'contribution': contribution,
        'sign': np.sign(contribution),
        'position': np.arange(len(selected.index) + 1, -1, -1),
        'label': explainer.label
    }

    return result, yhats


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
