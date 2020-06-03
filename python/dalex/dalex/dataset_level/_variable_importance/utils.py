import multiprocessing as mp

import numpy as np
import pandas as pd


def calculate_variable_importance(explainer,
                                  type,
                                  loss_function,
                                  variables,
                                  N,
                                  B,
                                  label,
                                  processes,
                                  keep_raw_permutations):
    if processes == 1:
        result = [None] * B
        for i in range(B):
            result[i] = loss_after_permutation(explainer.data, explainer.y, explainer.model, explainer.predict_function,
                                               loss_function, variables, N)
    else:
        pool = mp.Pool(processes)
        result = pool.starmap_async(loss_after_permutation, [
            (explainer.data, explainer.y, explainer.model, explainer.predict_function, loss_function, variables, N) for
            i in range(B)]).get()
        pool.close()

    raw = pd.concat(result, sort=True)
    result = raw.mean().sort_values().reset_index()
    result['label'] = label

    result.rename(columns={0: 'dropout_loss', 'index': 'variable'}, inplace=True)

    if type == "ratio":
        result.loc[:, 'dropout_loss'] = result.loc[:, 'dropout_loss'] / result.loc[
            result.variable == '_full_model_', 'dropout_loss'].values

    if type == "difference":
        result.loc[:, 'dropout_loss'] = result.loc[:, 'dropout_loss'] - result.loc[
            result.variable == '_full_model_', 'dropout_loss'].values

    raw_permutations = raw.reset_index(drop=True) if keep_raw_permutations else None

    return result, raw_permutations


def loss_after_permutation(data, y, model, predict, loss_function, variables, N):
    if N is None:
        N = data.shape[0]
    else:
        N = min(N, data.shape[0])

    sampled_rows = np.random.choice(np.arange(N), N, replace=False)

    sampled_data = data.iloc[sampled_rows, :]

    observed = y[sampled_rows]

    # loss on the full model or when outcomes are permuted
    loss_full = loss_function(observed, predict(model, sampled_data))

    sampled_rows2 = np.random.choice(range(observed.shape[0]), observed.shape[0], False)
    loss_baseline = loss_function(observed[sampled_rows2], predict(model, sampled_data))

    loss_features = {}
    for variables_set_key in variables:
        ndf = sampled_data.copy()
        ndf.loc[:, variables[variables_set_key]] = ndf.iloc[
                                                   np.random.choice(range(ndf.shape[0]), ndf.shape[0], False), :].loc[:,
                                                   variables[variables_set_key]].values

        predicted = predict(model, ndf)

        loss_features[variables_set_key] = loss_function(observed, predicted)

    loss_features['_full_model_'] = loss_full
    loss_features['_baseline_'] = loss_baseline

    return pd.DataFrame(loss_features, index=[0])
