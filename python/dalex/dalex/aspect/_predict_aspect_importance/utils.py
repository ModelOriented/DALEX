"""Legacy pandas-returning adapters over native Aspect importance kernels."""
import numpy as np
from ..._dataframe import observation_values
from dalex import _global_checks


def _project(explainer, new_observation, variable_groups, **options):
    from .._native_importance import calculate
    from .._table import AspectTable
    data, names, value_lists = calculate(explainer, new_observation, variable_groups, **options)
    return AspectTable(data, names, [None] * len(names), new_observation,
        value_lists=value_lists, columns=['aspect_name', 'variable_names', 'variable_values',
                                          'importance', 'label'])._dalex_pandas_view()


def calculate_predict_aspect_importance(explainer, new_observation, variable_groups,
                                       N, n_aspects, sample_method, f, random_state):
    _global_checks.global_check_import('scikit-learn', 'Predict Aspect Importance explanations')
    return _project(explainer, new_observation, variable_groups, kind='default', N=N,
                    n_aspects=n_aspects, sample_method=sample_method, f=f, random_state=random_state)


def calculate_shap_predict_aspect_importance(explainer, new_observation, variable_groups,
                                            N, B, processes, random_state):
    return _project(explainer, new_observation, variable_groups, kind='shap', N=N, B=B,
                    processes=processes, random_state=random_state)


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


def iterate_paths(predict_function, data_sample, new_observation, variable_groups, p, b, rng, values=None):
    random_path = rng.choice(p, p, replace=False)
    return get_single_random_path(predict_function, data_sample, new_observation,
                                  variable_groups, random_path, b, values)


def get_single_random_path(predict_function, data_sample, new_observation,
                           variable_groups, random_path, b, values=None):
    from .._native_importance import path_contributions
    from narwhals.dependencies import get_pandas
    values = observation_values(new_observation) if values is None else values
    groups, names = list(variable_groups.values()), list(variable_groups)
    changes = path_contributions(predict_function, data_sample, values, groups,
                                  random_path, reset_index=True)
    # This private helper has historically returned pandas. Only publication
    # happens here; native model inputs and numeric changes use shared kernels.
    return get_pandas().DataFrame({'aspect_name': [names[i] for i in random_path],
        'variable_names': [groups[i] for i in random_path], 'importance': changes, 'B': b})
