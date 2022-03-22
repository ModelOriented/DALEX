from copy import deepcopy

import numpy as np
import pandas as pd

from . import checks
from ... import _global_checks
from ..._explainer import helper
from ...model_explanations._model_performance import utils


# -------------- Objects needed in creation of GroupFairnessClassification object in object.py --------------

class ConfusionMatrix:

    def __init__(self, y_true, y_pred, cutoff):
        assert len(y_true) == len(y_pred)
        assert 0 < cutoff < 1

        self.cutoff = cutoff
        self.tp = ((y_true == 1) * (y_pred >= self.cutoff)).sum()
        self.fp = ((y_true == 0) * (y_pred >= self.cutoff)).sum()
        self.tn = ((y_true == 0) * (y_pred < self.cutoff)).sum()
        self.fn = ((y_true == 1) * (y_pred < self.cutoff)).sum()


class SubgroupConfusionMatrix:

    def __init__(self, y_true, y_pred, protected, cutoff):
        assert len(y_true) == len(y_pred) == len(protected)
        assert isinstance(cutoff, dict)

        subgroups = np.unique(protected)
        sub_dict = {}

        for sub in subgroups:
            sub_indexes = np.where(protected == sub)
            sub_y_true = y_true[sub_indexes]
            sub_y_pred = y_pred[sub_indexes]

            sub_dict[sub] = ConfusionMatrix(sub_y_true, sub_y_pred, cutoff.get(sub))
        self.sub_dict = sub_dict


class SubgroupConfusionMatrixMetrics:
    """Calculate confusion matrix metrics for each subgroup

    Parameters
    -----------
    sub_confusion_matrix : SubgroupConfusionMatrix
        Object with calculated confusion matrix values for each subgroup

    Attributes
    -----------
    subgroup_confusion_matrix_metrics : dict
        Dictionary with confusion matrix metrics for each subgroup
    """

    def __init__(self, sub_confusion_matrix):
        assert isinstance(sub_confusion_matrix, SubgroupConfusionMatrix)

        matrix_dict = sub_confusion_matrix.sub_dict
        subgroup_confusion_matrix_metrics = {}

        for sub, cm in matrix_dict.items():
            tp, tn, fp, fn = cm.tp, cm.tn, cm.fp, cm.fn

            TPR = TNR = PPV = NPV = FNR = FPR = FDR = FOR = ACC = STP = np.nan

            if tp + fn > 0:
                TPR = tp / (tp + fn)
                FNR = fn / (tp + fn)
            if tn + fp > 0:
                TNR = tn / (tn + fp)
                FPR = fp / (fp + tn)
            if tp + fp > 0:
                PPV = tp / (tp + fp)
                FDR = fp / (tp + fp)
            if tn + fn > 0:
                NPV = tn / (tn + fn)
                FOR = fn / (tn + fn)
            if fp + tp + fn + tn > 0:
                ACC = (tp + tn) / (tp + tn + fp + fn)
                STP = (tp + fp) / (tp + tn + fp + fn)

            cf_metrics = {'TPR': TPR, 'TNR': TNR, 'PPV': PPV, 'NPV': NPV,
                          'FNR': FNR, 'FPR': FPR, "FDR": FDR, 'FOR': FOR, 'ACC': ACC, 'STP': STP}

            for metric in cf_metrics.keys():
                if not pd.isna(cf_metrics.get(metric)):
                    cf_metrics[metric] = round(cf_metrics.get(metric), 3)

            subgroup_confusion_matrix_metrics[sub] = deepcopy(cf_metrics)

        self.subgroup_confusion_matrix_metrics = subgroup_confusion_matrix_metrics

    def to_vertical_DataFrame(self) -> pd.DataFrame:

        columns = ['metric', 'subgroup', 'score']
        data = pd.DataFrame(columns=columns)
        metrics = self.subgroup_confusion_matrix_metrics
        for subgroup in metrics.keys():
            metric = metrics.get(subgroup)
            subgroup_vec = np.repeat(subgroup, len(metric))
            sub_df = pd.DataFrame({'metric': metric.keys(), 'subgroup': subgroup_vec, 'score': metric.values()})
            data = pd.concat([data, sub_df])
        return data

    def to_horizontal_DataFrame(self) -> pd.DataFrame:

        metrics = self.subgroup_confusion_matrix_metrics
        return pd.DataFrame(metrics).transpose()

    def __str__(self):

        return f'Object _SubgroupConfusionMatrixMetrics was converted` to data frame, top rows: \n' \
               f'{self.to_vertical_DataFrame().head().to_string()}'


# -------------- Functions needed in creation and methods of GroupFairnessClassification object in object.py --------------

def calculate_parity_loss(sub_confusion_matrix_metrics, privileged):
    """
    Calculates parity_loss with formula
    M_parity_loss = sum(|log(M/M_p)|)

    where
    M - vector of metrics for each subgroup
    M_p - value in metric for privileged subgroup
    """
    assert isinstance(sub_confusion_matrix_metrics, SubgroupConfusionMatrixMetrics)
    df_ratio = calculate_ratio(sub_confusion_matrix_metrics, privileged)
    columns = df_ratio.columns
    df_log = np.log(df_ratio.to_numpy())
    df_absolute_log = abs(df_log)
    df_summed = pd.DataFrame(df_absolute_log, index=df_ratio.index.values, columns=columns).apply(sum)
    return df_summed


def calculate_ratio(sub_confusion_matrix_metrics, privileged):
    """
    Calculates ratio of metrics - divides all by privileged
    Does not allow for zeros in ratios, instead puts NaN
    """
    assert isinstance(sub_confusion_matrix_metrics, SubgroupConfusionMatrixMetrics)
    df = sub_confusion_matrix_metrics.to_horizontal_DataFrame()

    privileged_index = np.where(df.index.values == privileged)[0][0]
    df_ratio = df / df.iloc[privileged_index, :]
    df_ratio = df_ratio.to_numpy()
    df_ratio[df_ratio == 0] = np.nan
    df_ratio[np.isinf(df_ratio)] = np.nan
    df_out = pd.DataFrame(df_ratio, index=df.index.values, columns=df.columns)
    return df_out


# -------------- Functions used in plots --------------
def _unwrap_parity_loss_data(fobject, other_objects, metrics, verbose):
    """ Unwrap parity loss data
    Some functions use using same computations, they are put here.
    Function creates parity loss DataFrame from fobject and stacks it with
    data from other_objects. This is private, helper function.
    """

    data = pd.DataFrame()
    data['score'] = deepcopy(fobject.parity_loss)
    data['label'] = np.repeat(fobject.label, len(fobject.parity_loss))
    data['metric'] = data.index
    data = data.reset_index(drop=True)

    if other_objects is not None:
        for object in other_objects:
            other_data = pd.DataFrame()
            other_data['score'] = deepcopy(object.parity_loss)
            other_data['label'] = np.repeat(object.label, len(object.parity_loss))
            other_data['metric'] = other_data.index
            other_data = other_data.reset_index(drop=True)
            data = pd.concat([data, other_data])

    if len(metrics) > 1:
        data = data.loc[data.metric.isin(metrics), :]
    else:
        data = data.loc[data.metric == metrics[0]]
    # checking for nans
    if any(pd.isna(data.score)):
        models_with_nans = set(data.loc[pd.isna(data.score), :].label)
        helper.verbose_cat(f"Found NaNs in following models: {models_with_nans}", verbose)

    return data


def _fairness_theme(title):
    return {'title_text': title,
            'template': 'plotly_white',
            'title_x': 0.5,
            'title_y': 0.99,
            'titlefont': {'size': 25},
            'font': {'color': "#371ea3"},
            'margin': {'t': 78, 'b': 71, 'r': 30}}


def _metric_ratios_2_df(fobject):
    """
    Converts GroupFairnessClassification
    to elegant DataFrame with 4 columns (subgroup, metric, score, label)
    """

    data = fobject.result
    data = data.stack(dropna=False)
    data = data.reset_index()
    data.columns = ["subgroup", "metric", "score"]
    data = data.loc[data.metric.isin(["TPR", "ACC", "PPV", "FPR", "STP"])]
    data = data.loc[data.subgroup != fobject.privileged]
    data.score -= 1

    data['label'] = np.repeat(fobject.label, data.shape[0])

    return data


def _classification_performance(fobject, verbose, type='accuracy'):
    tp = tn = fp = fn = 0
    for key, val in fobject._subgroup_confusion_matrix.sub_dict.items():
        tp += val.tp
        tn += val.tn
        fp += val.fp
        fn += val.fn

    if type == 'accuracy':
        return utils.accuracy(tp, fp, tn, fn)
    if type == 'auc':
        helper.verbose_cat("Beware, that auc metric is insensitive to cutoffs", verbose)
        return utils.auc(fobject.y_hat, fobject.y)
    if type == 'recall':
        return utils.recall(tp, fp, tn, fn)
    if type == 'precision':
        return utils.precision(tp, fp, tn, fn)
    if type == 'f1':
        return utils.f1(tp, fp, tn, fn)
    else:
        raise TypeError(f'type \'{type}\' not supported')


# -------------- Functions needed in creation and methods of GroupFairnessRegression object in object.py --------------

class RegressionDict:
    # @TODO - implement metrics based on performance measure
    def __init__(self, y, y_hat, protected, privileged, verbose=False):
        self.regression_dict = {}
        self.subgroup_metrics = {}
        self.subgroup_metric_comparison = {}

        subgroups = set(protected)

        for subgroup in subgroups:
            sub_y = y[np.where(protected == subgroup)]
            sub_y_hat = y_hat[np.where(protected == subgroup)]
            self.regression_dict[subgroup] = {'y': sub_y, 'y_hat': sub_y_hat}

            self.subgroup_metrics[subgroup] = {
                'mean_error': (sum(sub_y_hat - sub_y)) / (len(y)),
                'mae': utils.mae(sub_y_hat, sub_y),
                'rmse': utils.rmse(sub_y_hat, sub_y),
                'mean_prediction': np.mean(sub_y_hat)
            }

            self.subgroup_metric_comparison[subgroup] = {
                'mae_ratio': self.subgroup_metrics[subgroup].get("mae") / self.subgroup_metrics[privileged].get("mae"),
                'rmse_ratio': self.subgroup_metrics[subgroup].get("rmse") / self.subgroup_metrics[privileged].get(
                    "rmse"),
                'mean_prediction_ratio': self.subgroup_metrics[subgroup].get("mean_prediction") / self.subgroup_metrics[
                    privileged].get("mean_prediction"),
            }

    def __str__(self):
        return f"Fairness in Regression Dictionary\nSubgroup Metrics: {self.subgroup_metrics.values()}\n" \
               f"Subgroup Metric Comparison: {self.subgroup_metric_comparison.values()}"


# -------------- Functions needed in creation and methods of GroupFairnessRegression object in object.py --------------

def calculate_regression_measures(y, y_hat, protected, privileged):
    _global_checks.global_check_import('scikit-learn', 'fairness in regression')
    from sklearn.linear_model import LogisticRegression

    unique_protected = np.unique(protected)
    unique_unprivileged = unique_protected[unique_protected != privileged]

    data = pd.DataFrame(columns=['subgroup', 'independence', 'separation', 'sufficiency'])

    for unprivileged in unique_unprivileged:
        # filter elements
        array_elements = np.isin(protected, [privileged, unprivileged])

        y_u = ((y[array_elements] - y[array_elements].mean()) / y[array_elements].std()).reshape(-1, 1)
        s_u = ((y_hat[array_elements] - y_hat[array_elements].mean()) / y_hat[array_elements].std()).reshape(-1, 1)

        a = np.where(protected[array_elements] == privileged, 1, 0)

        p_s = LogisticRegression()
        p_ys = LogisticRegression()
        p_y = LogisticRegression()

        p_s.fit(s_u, a)
        p_y.fit(y_u, a)
        p_ys.fit(np.c_[y_u, s_u], a)

        pred_p_s = p_s.predict_proba(s_u.reshape(-1, 1))[:, 1]
        pred_p_y = p_y.predict_proba(y_u.reshape(-1, 1))[:, 1]
        pred_p_ys = p_ys.predict_proba(np.c_[y_u, s_u])[:, 1]

        n = len(a)

        r_ind = ((n - a.sum()) / a.sum()) * (pred_p_s / (1 - pred_p_s)).mean()
        r_sep = ((pred_p_ys / (1 - pred_p_ys) * (1 - pred_p_y) / pred_p_y)).mean()
        r_suf = ((pred_p_ys / (1 - pred_p_ys)) * ((1 - pred_p_s) / pred_p_s)).mean()

        to_append = pd.DataFrame({'subgroup': [unprivileged],
                                  'independence': [r_ind],
                                  'separation': [r_sep],
                                  'sufficiency': [r_suf]})

        data = pd.concat([data, to_append])

    # append the scale
    to_append = pd.DataFrame({'subgroup': [privileged],
                              'independence': [1],
                              'separation': [1],
                              'sufficiency': [1]})
    ## TODO: this should be uncommented but adds blanks to the plots
    # data = pd.concat([data, to_append]) 

    data.index = data.subgroup
    data = data.iloc[:, 1:]
    return data


# -------------- Helper functions --------------
def fairness_check_metrics():
    return ["TPR", "ACC", "PPV", "FPR", "STP"]


def universal_fairness_check(self, epsilon, verbose, num_for_not_fair, num_for_no_decision, metrics):
    if epsilon is None:
        epsilon = self.epsilon
    else:
        epsilon = checks.check_epsilon(epsilon)

    metric_ratios = self.result

    subgroups = np.unique(self.protected)
    subgroups_without_privileged = subgroups[subgroups != self.privileged]
    metric_ratios = metric_ratios.loc[subgroups_without_privileged, metrics]

    metrics_exceeded = ((metric_ratios > 1 / epsilon) | (epsilon > metric_ratios)).apply(sum, 0)

    names_of_exceeded_metrics = list(metrics_exceeded.index[metrics_exceeded != 0])
    if len(names_of_exceeded_metrics) >= num_for_not_fair:
        print(f'Bias detected in {len(names_of_exceeded_metrics)} metrics: {", ".join(names_of_exceeded_metrics)}')
    elif len(names_of_exceeded_metrics) == num_for_no_decision:
        print(f'Bias detected in {len(names_of_exceeded_metrics)} metric: {names_of_exceeded_metrics[0]}')
    else:
        print("No bias was detected!")

    # arbitrary decision
    if len(names_of_exceeded_metrics) >= num_for_not_fair:
        conclusion = 'is not fair because 2 or more criteria exceeded acceptable limits set by epsilon'
    elif len(names_of_exceeded_metrics) == num_for_no_decision:
        conclusion = 'cannot be called fair because 1 criterion exceeded acceptable limits set by epsilon.\n' \
                     'It does not mean that your model is unfair ' \
                     'but it cannot be automatically approved based on these metrics'
    else:
        conclusion = 'is fair in terms of checked fairness criteria'

    print(f'\nConclusion: your model {conclusion}.')

    print(
        f'\nRatios of metrics, based on \'{self.privileged}\'. Parameter \'epsilon\' was set to {epsilon}'
        f' and therefore metrics should be within ({epsilon}, {round(1 / epsilon, 3)})')
    print(metric_ratios.to_string())
    if pd.isna(metric_ratios).sum().sum() > 0:
        helper.verbose_cat(
            '\nWarning!\nTake into consideration that NaN\'s are present, consider checking \'metric_scores\' '
            'plot to see the difference', verbose=verbose)


def get_nice_ticks(min_value, max_value, max_ticks=9):
    tick_range = readable_number(max_value - min_value, False)
    tick_spacing = readable_number(tick_range / (max_ticks - 1), True)
    readable_minimum = np.floor(min_value / tick_spacing) * tick_spacing
    readable_maximum = np.ceil(max_value / tick_spacing) * tick_spacing
    return readable_minimum, readable_maximum, tick_spacing


def readable_number(tick_range, round_number):
    exponent = np.floor(np.log10(tick_range))
    fraction = tick_range / np.power(10, exponent)

    if round_number:
        if fraction < 1.5:
            readable_tick = 1
        elif fraction < 3:
            readable_tick = 2
        elif fraction < 7:
            readable_tick = 5
        else:
            readable_tick = 10
    else:
        if fraction <= 1:
            readable_tick = 1
        elif fraction <= 2:
            readable_tick = 2
        elif fraction <= 5:
            readable_tick = 5
        else:
            readable_tick = 10

    return readable_tick * np.power(10, exponent)






