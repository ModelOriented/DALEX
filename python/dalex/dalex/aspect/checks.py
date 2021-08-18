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
    if value < -0.01 or value > 1.01:
        warnings.warn(
            "Association value outside the range [0, 1]. Truncated to the range."
        )

    if value < 0:
        value = 0

    if value > 1:
        value = 1

    return value


def check_method_loss_function(explainer, loss_function):
    if (
        loss_function is not None or explainer.model_type is None
    ):  # user passed a function or type is not known
        return loss_function
    elif explainer.model_type == "regression":
        return "rmse"
    elif explainer.model_type == "classification":
        return "1-auc"
