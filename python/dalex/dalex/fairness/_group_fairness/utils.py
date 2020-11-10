from copy import deepcopy
import numpy as np
import pandas as pd
from dalex._explainer.helper import verbose_cat
from dalex.dataset_level._model_performance.utils import *
# -------------- Objects needed in creation of object in object.py --------------

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

            TNR = PPV = NPV = FNR = FPR = FDR = FOR = ACC = STP = np.nan

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

        columns = ['metric', 'subgroup', 'score']
        data = pd.DataFrame(columns=columns)
        metrics = self.subgroup_confusion_matrix_metrics
        for subgroup in metrics.keys():
            metric = metrics.get(subgroup)
            subgroup_vec = np.repeat(subgroup, len(metric))
            sub_df = pd.DataFrame({'metric': metric.keys(), 'subgroup': subgroup_vec, 'score': metric.values()})
            data = data.append(sub_df)
        return data

    def to_horizontal_DataFrame(self) -> pd.DataFrame:

        metrics = self.subgroup_confusion_matrix_metrics
        return pd.DataFrame(metrics).transpose()

    def __str__(self):

        return f'Object _SubgroupConfusionMatrixMetrics was converted` to data frame, top rows: \n' \
               f'{self.to_vertical_DataFrame().head().to_string()}'


# -------------- Functions needed in creation and methods of object in object.py --------------

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
    Calculates ratio of metrix - divides all by privileged
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
            data = data.append(other_data)

    if len(metrics) > 1:
        data = data.loc[data.metric.isin(metrics), :]
    else:
        data = data.loc[data.metric == metrics[0]]
    # checking for nans
    if any(np.isnan(data.score)):
        models_with_nans = set(data.loc[np.isnan(data.score), :].label)
        verbose_cat(f"Found NaNs in following models: {models_with_nans}", verbose)

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
    data = data.stack()
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

    if type=='accuracy':
        return accuracy(tp, fp, tn, fn)
    if type=='auc':
        verbose_cat("Beware, that auc metric is insensitive to cutoffs", verbose)
        return auc(fobject.y_hat, fobject.y)
    if type=='recall':
        return recall(tp, fp, tn, fn)
    if type=='precision':
        return  precision(tp, fp, tn, fn)
    if type=='f1':
        return f1(tp, fp, tn, fn)
    else:
        raise TypeError(f'type \'{type}\' not supported')

# -------------- Helper functions --------------
def fairness_check_metrics():
    return ["TPR", "ACC", "PPV", "FPR", "STP"]