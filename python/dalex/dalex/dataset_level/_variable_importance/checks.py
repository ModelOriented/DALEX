from .._model_performance.utils import *


def check_loss_function(loss_function):
    loss_functions = {
        'rmse': rmse,
        'mse': mse,
        'mae': mae,
        'mad': mad,
        'r2': r2,
        'auc': auc
    }
    if isinstance(loss_function, str):
        loss_function = loss_functions[loss_function]

    return loss_function


def check_variable_groups(variable_groups, explainer):
    if variable_groups is not None:
        if not isinstance(variable_groups, dict):
            raise TypeError("variable_groups should be of class dict")

        wrong_names = np.empty(len(variable_groups))
        for i, key in enumerate(variable_groups):
            if isinstance(variable_groups[key], list):
                variable_groups[key] = np.array(variable_groups[key])
            elif not isinstance(variable_groups[key], np.ndarray):
                raise TypeError("variable_groups' is a dict of lists of variables")

            if not isinstance(variable_groups[key][0], str):
                raise TypeError("variable_groups' is a dict of lists of variables")

            wrong_names[i] = np.in1d(variable_groups[key], explainer.data.columns).all()

        wrong_names = not wrong_names.all()

        if wrong_names:
            raise ValueError("You have passed wrong variables names in variable_groups argument")

    return variable_groups


def check_B(B):
    return max(1, np.round(B))


def check_variables(variables, variable_groups, explainer):
    if variable_groups is None:
        # if `variables` are not specified, then extract from data
        if variables is None:
            variables = list(explainer.data.columns)
        elif not isinstance(variables, (list, np.ndarray)):
            raise TypeError("variables must be list or numpy.ndarray or a dict")
    else:
        if variables is not None:
            raise Warning("Variables parameter ignored, taken variable_groups instead.")
        variables = variable_groups

    if not isinstance(variables, (list, np.ndarray, dict)):
        raise TypeError("variables must be list or numpy.ndarray or a dict")

    if isinstance(variables, (list, np.ndarray)):
        result = {}
        for var in variables:
            result[var] = var

        variables = result

    return variables


def check_label(label, explainer):
    if label is None:
        label = explainer.label

    return label


def check_type(type):
    if isinstance(type, tuple):
        type = type[0]

    if not isinstance(type, (str,)):
        raise TypeError("type must be a string")

    if type not in ['variable_importance', 'ratio', 'difference']:
        raise ValueError("type must be 'variable_importance'/'ratio'/'difference'")

    return type


def check_random_state(random_state):
    if random_state is not None:
        np.random.set_state(random_state)

    return random_state


def check_keep_raw_permutations(keep_raw_permutations, B):
    return B > 1 if keep_raw_permutations is None else keep_raw_permutations
