from copy import deepcopy
import numpy as np
import warnings

def check_variable_groups(variable_groups, explainer):
    if variable_groups is not None:
        if not isinstance(variable_groups, dict):
            raise TypeError("'variable_groups' should be of class dict")

        wrong_names = np.empty(len(variable_groups))
        for i, key in enumerate(variable_groups):
            if isinstance(variable_groups[key], list):
                variable_groups[key] = np.array(variable_groups[key])
            elif not isinstance(variable_groups[key], np.ndarray):
                raise TypeError("'variable_groups' is a dict of lists of variables")

            if not isinstance(variable_groups[key][0], str):
                raise TypeError("'variable_groups' is a dict of lists of variables")

            wrong_names[i] = np.in1d(
                variable_groups[key],
                explainer.data.select_dtypes(include=np.number).columns,
            ).all()

        wrong_names = not wrong_names.all()

        if wrong_names:
            raise ValueError(
                "You have passed wrong variables names in variable_groups argument. "
                "'variable_groups' is a dict of lists of numeric variables."
            )

    return variable_groups


def check_assoc_value(value):
    if value < 0:
        value = 0

    if value > 1:
        value = 1

    return value


def check_method_loss_function(explainer, loss_function):
    if loss_function is not None:
        # user passed a function
        return loss_function
    elif explainer.model_type is None:
        # type is not known
        return "rmse"
    elif explainer.model_type == "regression":
        return "rmse"
    elif explainer.model_type == "classification":
        return "1-auc"

    
def check_method_depend(depend_method, corr_method, agg_method):
    depend_method_types = ('assoc', 'pps')
    depend_method_aliases = {'association': 'assoc', "PPS": 'pps', 'stats': 'assoc'}
    corr_method_types = ('spearman', 'pearson', 'kendall')
    agg_method_types = ('max', 'min', 'mean')
    agg_method_aliases = {'maximum': 'max', 'minimum': 'min', 'avg': 'mean', 'average': 'mean'}
    if isinstance(depend_method, str):
        if depend_method not in depend_method_types:
            if depend_method not in depend_method_aliases:
                raise ValueError("'depend_method' must be one of: {}".format(', '.join(depend_method_types+tuple(depend_method_aliases))))
            else:
                depend_method = depend_method_aliases[depend_method]
        if depend_method == "assoc":
            if corr_method not in corr_method_types:
                raise ValueError("'corr_method' must be one of: {}".format(', '.join(corr_method_types)))
        if depend_method == "pps":
            if agg_method not in agg_method_types:
                if agg_method not in agg_method_aliases:
                    raise ValueError("'agg_method' must be one of: {}".format(', '.join(agg_method_types)))
                else: 
                    agg_method = agg_method_aliases[agg_method]
    return depend_method, corr_method, agg_method

