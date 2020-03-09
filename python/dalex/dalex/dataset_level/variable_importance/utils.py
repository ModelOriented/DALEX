import numpy as np
import pandas as pd


def calculate_variable_importance(explainer,
                                  type,
                                  loss_function,
                                  variables,
                                  n_sample,
                                  B,
                                  label,
                                  keep_raw_permutations):
    result = [None] * B

    for i in range(B):
        result[i] = loss_after_permutation(explainer, loss_function, variables, n_sample)

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


def loss_after_permutation(explainer, loss_function, variables, n_sample):
    if n_sample is not None:
        sampled_rows = np.random.choice(range(explainer.data.shape[0]), n_sample, False)
    else:
        sampled_rows = np.arange(explainer.data.shape[0])

    sampled_data = explainer.data.iloc[sampled_rows, :]

    observed = explainer.y[sampled_rows]

    # loss on the full model or when outcomes are permuted
    loss_full = loss_function(observed, explainer.predict_function(explainer.model, sampled_data))

    sampled_rows2 = np.random.choice(range(observed.shape[0]), observed.shape[0], False)
    loss_baseline = loss_function(observed[sampled_rows2],
                                       explainer.predict_function(explainer.model, sampled_data))

    loss_features = {}
    for variables_set_key in variables:
        ndf = sampled_data.copy()
        ndf.loc[:, variables[variables_set_key]] = ndf.iloc[
            np.random.choice(range(ndf.shape[0]), ndf.shape[0], False), :].loc[:, variables[variables_set_key]].values

        predicted = explainer.predict_function(explainer.model, ndf)

        loss_features[variables_set_key] = loss_function(observed, predicted)

    loss_features['_full_model_'] = loss_full
    loss_features['_baseline_'] = loss_baseline

    return pd.DataFrame(loss_features, index=[0])
