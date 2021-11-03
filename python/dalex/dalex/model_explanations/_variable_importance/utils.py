import multiprocessing as mp
from numpy.random import SeedSequence, default_rng

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
                                  keep_raw_permutations,
                                  random_state):
    if processes == 1:
        result = [None] * B
        for i in range(B):
            result[i] = loss_after_permutation(explainer.data, explainer.y, explainer.model, explainer.predict_function,
                                               loss_function, variables, N, np.random)
    else:
        # Create number generator for each iteration
        ss = SeedSequence(random_state)
        generators = [default_rng(s) for s in ss.spawn(B)]
        pool = mp.get_context('spawn').Pool(processes)
        result = pool.starmap_async(loss_after_permutation, [
            (explainer.data, explainer.y, explainer.model, explainer.predict_function, loss_function, variables, N, generators[i]) for
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


def loss_after_permutation(data, y, model, predict, loss_function, variables, N, rng):
    if isinstance(N, int):
        N = min(N, data.shape[0])
        sampled_rows = rng.choice(np.arange(data.shape[0]), N, replace=False)
        sampled_data = data.iloc[sampled_rows, :]
        observed = y[sampled_rows]
    else:
        sampled_data = data
        observed = y
    
    # loss on the full model or when outcomes are permuted
    loss_full = loss_function(observed, predict(model, sampled_data))

    sampled_rows2 = rng.choice(range(observed.shape[0]), observed.shape[0], replace=False)
    loss_baseline = loss_function(observed[sampled_rows2], predict(model, sampled_data))

    loss_features = {}
    for variables_set_key in variables:
        ndf = sampled_data.copy()
        ndf.loc[:, variables[variables_set_key]] = ndf.iloc[
                                                   rng.choice(range(ndf.shape[0]), ndf.shape[0], False), :].loc[:,
                                                   variables[variables_set_key]].values

        predicted = predict(model, ndf)

        loss_features[variables_set_key] = loss_function(observed, predicted)

    loss_features['_full_model_'] = loss_full
    loss_features['_baseline_'] = loss_baseline

    return pd.DataFrame(loss_features, index=[0])
