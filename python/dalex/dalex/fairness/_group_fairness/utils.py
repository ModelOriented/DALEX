from copy import deepcopy

import numpy as np
import pandas as pd
import narwhals.stable.v2 as nw
from ._table import FairnessTable, ParityTable

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
                if not np.isnan(cf_metrics.get(metric)):
                    cf_metrics[metric] = round(cf_metrics.get(metric), 3)

            subgroup_confusion_matrix_metrics[sub] = deepcopy(cf_metrics)

        self.subgroup_confusion_matrix_metrics = subgroup_confusion_matrix_metrics

    def to_vertical_DataFrame(self) -> pd.DataFrame:
        from ._plotting import long_frame
        frame = nw.from_native(self.to_native_table('pandas').data)
        return long_frame(frame).select('metric', 'subgroup', 'score').to_pandas().reset_index(drop=True)

    def to_horizontal_DataFrame(self) -> pd.DataFrame:
        return self.to_native_table('pandas')._dalex_pandas_view()

    def to_native_table(self, backend='pandas'):
        metrics = self.subgroup_confusion_matrix_metrics
        columns = list(next(iter(metrics.values())))
        values = {'subgroup': list(metrics)}
        values.update({name: np.asarray([row[name] for row in metrics.values()], dtype=float) for name in columns})
        return FairnessTable(nw.from_dict(values, backend=backend).to_native())

    def __str__(self):

        return f'Object _SubgroupConfusionMatrixMetrics was converted` to data frame, top rows: \n' \
               f'{self.to_vertical_DataFrame().head().to_string()}'


# -------------- Functions needed in creation and methods of GroupFairnessClassification object in object.py --------------

def calculate_parity_loss(sub_confusion_matrix_metrics, privileged, *, native=False, backend='pandas'):
    table = calculate_ratio(sub_confusion_matrix_metrics, privileged, native=True, backend=backend)
    frame = nw.from_native(table.data, eager_only=True).drop('subgroup')
    values = np.sum(np.abs(np.log(frame.to_numpy())), axis=0)
    result = ParityTable(nw.from_dict({'metric': frame.columns, 'score': values}, backend=backend).to_native())
    return result if native else result._dalex_pandas_view()


def calculate_ratio(sub_confusion_matrix_metrics, privileged, *, native=False, backend='pandas'):
    assert isinstance(sub_confusion_matrix_metrics, SubgroupConfusionMatrixMetrics)
    rows = sub_confusion_matrix_metrics.subgroup_confusion_matrix_metrics
    groups = list(rows)
    columns = list(next(iter(rows.values())))
    matrix = np.asarray([[row[name] for name in columns] for row in rows.values()], dtype=float)
    values = matrix / matrix[groups.index(privileged)]
    values[(values == 0) | np.isinf(values)] = np.nan
    result = FairnessTable(nw.from_dict({'subgroup': groups, **{name: values[:, i] for i, name in enumerate(columns)}},
                                        backend=backend).to_native())
    return result if native else result._dalex_pandas_view()


# -------------- Functions used in plots --------------
def _unwrap_parity_loss_data(fobject, other_objects, metrics, verbose):
    """Legacy pandas adapter over the native parity plot table."""
    from ._plotting import parity_data, table_frame
    result = parity_data(fobject, other_objects, metrics, verbose).to_pandas()
    positions = [np.flatnonzero(np.isin(table_frame(obj, 'parity_loss')['metric'].to_list(), metrics))
                 for obj in [fobject, *(other_objects or [])]]
    result = result.loc[:, ['score', 'label', 'metric']]
    result.index = np.concatenate(positions)
    return result


def _fairness_theme(title):
    return {'title_text': title,
            'template': 'plotly_white',
            'title_x': 0.5,
            'title_y': 0.99,
            'title_font': {'size': 25},
            'font': {'color': "#371ea3"},
            'margin': {'t': 78, 'b': 71, 'r': 30}}


def _metric_ratios_2_df(fobject):
    """Legacy pandas adapter over the common native long-table kernel."""
    from ._plotting import long_table, table_frame
    metrics = ['TPR', 'ACC', 'PPV', 'FPR', 'STP']
    frame = table_frame(fobject)
    columns = [name for name in frame.columns if name != 'subgroup']
    selected = np.tile(np.isin(columns, metrics), len(frame)) & np.repeat(
        np.asarray(frame['subgroup'].to_list()) != fobject.privileged, len(columns))
    result = (long_table(fobject).filter(nw.col('metric').is_in(metrics) &
                                       (nw.col('subgroup') != fobject.privileged))
              .with_columns((nw.col('score') - 1).alias('score')).to_pandas())
    result.index = np.flatnonzero(selected)
    return result


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

def calculate_regression_measures(y, y_hat, protected, privileged, *, native=False, backend='pandas'):
    _global_checks.global_check_import('scikit-learn', 'fairness in regression')
    from sklearn.linear_model import LogisticRegression

    unique_protected = np.unique(protected)
    unique_unprivileged = unique_protected[unique_protected != privileged]

    data_list = []
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

        data_list.append((unprivileged, r_ind, r_sep, r_suf))
    columns = ['subgroup', 'independence', 'separation', 'sufficiency']
    values = {name: [row[i] for row in data_list] if i == 0 else np.asarray([row[i] for row in data_list], dtype=float)
              for i, name in enumerate(columns)}
    data = nw.from_dict(values, backend=backend).to_native()
    result = FairnessTable(data, index_name='subgroup', empty_objects=True)
    return result if native else result._dalex_pandas_view()



# -------------- Helper functions --------------
def fairness_check_metrics():
    return ["TPR", "ACC", "PPV", "FPR", "STP"]


def _fairness_check_table(obj, subgroups, metrics):
    from ._plotting import table_frame
    frame = table_frame(obj).with_row_index('__source_row__')
    missing_groups = set(subgroups) - set(frame['subgroup'].to_list())
    missing_metrics = set(metrics) - set(frame.columns)
    if missing_groups or missing_metrics:
        raise KeyError(list(missing_groups | missing_metrics))
    if not len(subgroups):
        return frame.head(0).select('subgroup', *metrics)
    order = nw.from_dict({'subgroup': list(subgroups), '__check_order__': np.arange(len(subgroups))},
                         backend=nw.get_native_namespace(frame))
    return (order.join(frame, on='subgroup', how='left')
            .sort(['__check_order__', '__source_row__']).select('subgroup', *metrics))


def _fairness_check_display(obj, frame, subgroups, metrics):
    """The historical pandas console renderer is an explicit output boundary."""
    from ..._dataframe import to_pandas
    value = obj.__dict__.get('_compatibility_result', obj.__dict__.get('result'))
    if isinstance(value, pd.DataFrame):
        # Caller edits retain their exact extension dtypes and index name.
        return value.loc[subgroups, metrics]
    result = to_pandas(frame.to_native()).set_index('subgroup')
    result.index.name = getattr(value, 'index_name', None)
    return result


def universal_fairness_check(self, epsilon, verbose, num_for_not_fair, num_for_no_decision, metrics):
    if epsilon is None:
        epsilon = self.epsilon
    else:
        epsilon = checks.check_epsilon(epsilon)

    subgroups = np.unique(self.protected)
    subgroups_without_privileged = subgroups[subgroups != self.privileged]
    frame = _fairness_check_table(self, subgroups_without_privileged, metrics)
    values = np.asarray(frame.select(metrics).to_numpy(), dtype=float)
    exceeded = ((values > 1 / epsilon) | (epsilon > values)).any(axis=0)
    legacy = self.__dict__.get('_compatibility_result', self.__dict__.get('result'))
    if isinstance(legacy, pd.DataFrame):
        # Python sum over nullable pandas comparisons historically propagates
        # pd.NA, and its object-result != 0 selection flags that metric. Preserve
        # those explicit legacy edits without coalescing NA and floating NaN.
        selected = legacy.loc[subgroups_without_privileged, metrics]
        for index, metric in enumerate(metrics):
            if isinstance(selected[metric].dtype, pd.api.extensions.ExtensionDtype):
                exceeded[index] |= any(value is pd.NA for value in selected[metric].array)
    names_of_exceeded_metrics = [metric for metric, flag in zip(metrics, exceeded) if flag]
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
    print(_fairness_check_display(self, frame, subgroups_without_privileged, metrics).to_string())
    if np.isnan(values).any():
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






