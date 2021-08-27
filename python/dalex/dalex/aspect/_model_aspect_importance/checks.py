from dalex.model_explanations._variable_importance.loss_functions import *
from dalex.model_explanations._model_performance.utils import *

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


def check_loss_function(loss_function):
    loss_functions = {
        'rmse': loss_root_mean_square,
        'mse': mse,
        'mae': mae,
        'mad': mad,
        'r2': r2,
        '1-auc': loss_one_minus_auc,
        'auc': loss_one_minus_auc  # semi backward compatibility
    }
    if isinstance(loss_function, str):
        loss_function = loss_functions[loss_function]

    return loss_function

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


