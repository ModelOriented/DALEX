import numpy as np
import pandas as pd
from copy import deepcopy
import multiprocessing as mp

from dalex import _global_checks


def calculate_predict_aspect_importance(
    explainer,
    new_observation,
    variable_groups,
    N,
    n_aspects,
    sample_method,
    f,
    random_state
):
    _global_checks.global_check_import('scikit-learn', 'Predict Aspect Importance explanations')
    # create sample from data and copy
    r = np.random.RandomState(random_state)
    N = min(explainer.data.shape[0], N)
    ids = r.randint(0, explainer.data.shape[0], N)
    n_sample = explainer.data.iloc[ids, :].copy()
    n_sample_changed = deepcopy(n_sample)
    n_sample_changed.reset_index(inplace=True, drop=True)

    if len(variable_groups) == 1:
        result = pd.DataFrame(
            {
                "aspect_name": list(variable_groups.keys())[0],
                "variable_names": [list(variable_groups.values())[0]],
                "variable_values": new_observation[list(variable_groups.values())[0]].values.tolist(),
                "importance": explainer.predict(new_observation) - explainer.y_hat.mean(),
                "label": explainer.label,
            }
        )
        return result

    # generate binary matrix for replacing aspect
    X_prim = get_sample(N, len(variable_groups), sample_method, f, random_state)

    # replace aspect in sample with observation's values of features
    for i in range(n_sample_changed.shape[0]):
        for k, variables_set_key in enumerate(variable_groups):
            if X_prim[i, k] == 1:
                variables = variable_groups[variables_set_key]
                n_sample_changed.loc[i, variables] = [
                    new_observation.iloc[0][var] for var in variables
                ]

    # calculate change in predictions
    y_changed = explainer.predict(n_sample_changed) - explainer.predict(n_sample)

    # fit linear model to binary matrix and change in predictions vector to estimate aspect importance
    X_prim = pd.DataFrame(X_prim)
    X_prim.columns = [aspect_name for aspect_name in variable_groups]

    if n_aspects == None:
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(X_prim, y_changed)
        model_coef = lr.coef_
    else:
        from sklearn.linear_model import lasso_path
        lasso_res_matrix = lasso_path(X_prim, y_changed)[1]
        indx = np.max(np.where(np.count_nonzero(lasso_res_matrix, axis=0) <= n_aspects))
        model_coef = lasso_res_matrix[:, indx]

    # generate result pd.DataFrame
    variable_names = [
        variable_groups[variables_set_key] for variables_set_key in variable_groups
    ]
    variable_values = [
        new_observation[variables_list].values[0] for variables_list in variable_names
    ]
    result = pd.DataFrame(
        {
            "aspect_name": X_prim.columns,
            "variable_names": variable_names,
            "variable_values": variable_values,
            "importance": model_coef,
            "label": explainer.label,
        }
    )
    result = result.sort_values(by="importance", key=abs, ascending=False).reset_index(
        drop=True
    )
    return result


def calculate_shap_predict_aspect_importance(
         explainer,
         new_observation,
         variable_groups,
         N,
         B,
         processes,
         random_state
):
    r = np.random.RandomState(random_state)
    N = min(explainer.data.shape[0], N)
    ids = r.randint(0, explainer.data.shape[0], N)
    data_sample = explainer.data.iloc[ids, :].copy()
    p = len(variable_groups)

    if processes == 1:
        result_list = [
            iterate_paths(explainer.predict, data_sample,
                           new_observation, variable_groups, p, b + 1, np.random)
            for b in range(B)]
    else:
        # create number generator for each iteration
        ss = np.random.SeedSequence(random_state)
        generators = [np.random.default_rng(s) for s in ss.spawn(B)]
        pool = mp.get_context('spawn').Pool(processes)
        result_list = pool.starmap_async(iterate_paths,
                                         [(explainer.predict, data_sample,
                                           new_observation, variable_groups, p, b + 1, generators[b]) for b in range(B)]).get()
        pool.close()

    tmp_result = pd.concat(result_list)

    # average over all of the paths
    variable_average = tmp_result.pivot(index='aspect_name', columns='B', values='importance').mean(axis=1)
    
    # generate result pd.DataFrame
    result = pd.DataFrame(
            {
            "aspect_name": variable_average.index.tolist(), 
            "importance": variable_average.values.tolist(),
            "label": explainer.label
            }
        )
    result["variable_names"] = result["aspect_name"].map(variable_groups)
    variable_values = [
        new_observation[variables_list].values[0] for variables_list in result["variable_names"]
    ]
    result["variable_values"] = variable_values
    result = result.sort_values(by="importance", key=abs, ascending=False).reset_index(
        drop=True
    )
    return result[["aspect_name", "variable_names", "variable_values", "importance", "label"]]


def get_sample(n, p, sample_method="default", f=2, random_state=None):
    X_prim = np.zeros(shape=(n, p))
    def_rng = np.random.default_rng(random_state)
    # f is average number of replaced zeros for binomial sampling
    if sample_method == "binom":
        if f <= 0: 
            # it must be > 0 
            raise ValueError("f should be positive")
        if f >= p: 
            # it must be < p (number_of_aspects)
            f = p - 1
        for i in range(n):
            n_of_changes = max(def_rng.binomial(p, f / p, 1), 1)
            X_prim[i, def_rng.choice(p, n_of_changes, replace=False)] = 1
    else:
        for i in range(n):
            X_prim[i, np.unique(def_rng.integers(0, p, 2))] = 1
    return X_prim


def iterate_paths(predict_function, data_sample, new_observation, variable_groups, p, b, rng):
    random_path = rng.choice(np.arange(p), p, replace=False)
    return get_single_random_path(predict_function, data_sample, new_observation, variable_groups, random_path, b)


def get_single_random_path(predict_function, data_sample, new_observation, variable_groups, random_path, b):
    current_data = deepcopy(data_sample)
    current_data.reset_index(inplace=True, drop=True)
    variable_groups_varnames = list(variable_groups.values())
    variable_groups_aspnames = list(variable_groups.keys())
    yhats = np.empty(len(random_path) + 1)
    yhats[0] = predict_function(current_data).mean()
    for i, candidate in enumerate(random_path):
        variables = variable_groups_varnames[candidate]
        for var in variables:
            current_data.loc[:, var] = new_observation.iloc[0][var]
        yhats[i + 1] = predict_function(current_data).mean()

    diffs = np.diff(yhats)
    variable_names = [variable_groups_varnames[i] for i in random_path]
    aspect_name = [variable_groups_aspnames[i] for i in random_path]

    return pd.DataFrame({
        'aspect_name': aspect_name,
        'variable_names': variable_names,
        'importance': diffs,
        'B': b
    })
