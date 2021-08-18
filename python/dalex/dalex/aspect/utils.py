import numpy as np
import pandas as pd
import ppscore as pps
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .. import _theme
from . import checks


def calculate_cat_assoc_matrix(data, categorical_variables, n):
    cat_assoc_matrix = pd.DataFrame(
        index=categorical_variables, columns=categorical_variables, dtype=float
    )
    for i, variable_name_a in enumerate(categorical_variables):
        for j, variable_name_b in enumerate(categorical_variables):
            if i > j:
                continue
            elif i == j:
                cramers_V = 1
            else:
                cont_tab = pd.crosstab(data[variable_name_a], data[variable_name_b])
                r = cont_tab.shape[0]
                c = cont_tab.shape[1]
                chi2_test = stats.chi2_contingency(cont_tab)
                phi = max(0, (chi2_test[0] / n) - (((r - 1) * (c - 1)) / (n - 1)))
                r = r - ((r - 1) ** 2 / (n - 1))
                c = c - ((c - 1) ** 2 / (n - 1))
                cramers_V = np.sqrt(phi / (min(c, r) - 1))
                cramers_V = checks.check_assoc_value(cramers_V)
            cat_assoc_matrix.loc[variable_name_a, variable_name_b] = cramers_V
            cat_assoc_matrix.loc[variable_name_b, variable_name_a] = cramers_V
    return cat_assoc_matrix


def calculate_cat_num_assoc_matrix(data, categorical_variables, numerical_variables, n):
    cat_num_assoc_matrix = pd.DataFrame(
        index=categorical_variables, columns=numerical_variables, dtype=float
    )
    for cat_variable_name in categorical_variables:
        for num_variable_name in numerical_variables:
            samples = [
                gr_data[num_variable_name].values
                for gr_name, gr_data in data.groupby(cat_variable_name, as_index=False)
            ]
            kruskal_wallis_test = stats.kruskal(*samples, nan_policy="omit")
            n_samples = len(samples)
            eta_squared = (kruskal_wallis_test.statistic - n_samples + 1) / (
                n - n_samples
            )
            eta_squared = checks.check_assoc_value(eta_squared)
            cat_num_assoc_matrix.loc[cat_variable_name, num_variable_name] = eta_squared
    return cat_num_assoc_matrix


def calculate_assoc_matrix(data, corr_method="spearman"):
    corr_matrix = data.corr(corr_method)
    numerical_variables = corr_matrix.columns
    categorical_variables = list(set(data.columns) - set(numerical_variables))
    n = len(data)
    cat_assoc_matrix = calculate_cat_assoc_matrix(data, categorical_variables, n)
    cat_num_assoc_matrix = calculate_cat_num_assoc_matrix(
        data, categorical_variables, numerical_variables, n
    )
    tmp_matrix = pd.concat([corr_matrix, cat_num_assoc_matrix.T], axis=1)
    tmp_matrix_2 = pd.concat([cat_num_assoc_matrix, cat_assoc_matrix], axis=1)
    assoc_matrix = pd.concat([tmp_matrix, tmp_matrix_2])
    return assoc_matrix


def calculate_pps_matrix(data, agg_method):
    pps_result = pps.matrix(data, sample=None)
    pps_matrix = pps_result[["x", "y", "ppscore"]].pivot(
        columns="x", index="y", values="ppscore"
    )
    if agg_method == "max":
        pps_matrix = np.maximum(pps_matrix, pps_matrix.transpose())
    if agg_method == "min":
        pps_matrix = np.minimum(pps_matrix, pps_matrix.transpose())
    if agg_method == "mean" or agg_method == "avg":
        pps_matrix = (pps_matrix + pps_matrix.transpose()) / 2
    return pps_matrix


def calculate_depend_matrix(
    data, depend_method="assoc", corr_method="spearman", agg_method="max"
):
    if depend_method == "assoc":
        depend_matrix = calculate_assoc_matrix(data, corr_method)
    if depend_method == "pps":
        depend_matrix = calculate_pps_matrix(data, agg_method)
    if callable(depend_method):
        try:
            depend_matrix = depend_method(data)
        except:
            raise ValueError(
                "You have passed wrong callable in depend_method argument. "
                "'depend_method' is the callable to use for calculating dependency matrix."
            )
    return depend_matrix


def calculate_linkage_matrix(depend_matrix, clust_method="complete"):
    dissimilarity = 1 - abs(depend_matrix)
    ## https://www.kaggle.com/sgalella/correlation-heatmaps-with-hierarchical-clustering?scriptVersionId=24045077&cellId=10
    linkage_matrix = linkage(squareform(dissimilarity), clust_method)

    return linkage_matrix

def get_dendrogram_aspects_ordered(hierarchical_clustering_dendrogram, depend_matrix):
    tick_dict = dict(
        zip(
            hierarchical_clustering_dendrogram.layout.yaxis.tickvals,
            [
                [var]
                for var in hierarchical_clustering_dendrogram.layout.yaxis.ticktext
            ],
        )
    )
    d = {k:v for v,k in enumerate(depend_matrix.columns)}
    _aspects_dendrogram_order = []
    for scatter in hierarchical_clustering_dendrogram.data:
        vars_list = tick_dict[scatter.y[1]] + tick_dict[scatter.y[2]]
        vars_list.sort(key=d.get)
        tick_dict[np.mean(scatter.y[1:3])] = vars_list
        _aspects_dendrogram_order.append(
            vars_list
        )
    
    _vars_min_depend, _min_depend = get_min_depend_from_matrix(depend_matrix, _aspects_dendrogram_order)

    return pd.DataFrame({
        "variables_names": _aspects_dendrogram_order,
        "min_depend": _min_depend,
        "vars_min_depend": _vars_min_depend
    })

def get_min_depend_from_matrix(depend_matrix, variables_list): 
    _vars_min_depend = []
    _min_depend = []
    for vars in variables_list:
        depend_submatrix = abs(depend_matrix).loc[list(vars), list(vars)]
        _min = np.array(depend_submatrix).min()
        _min_idx = np.where(depend_submatrix == _min)
        var_a_idx, var_b_idx = _min_idx[0][0], _min_idx[1][0]
        _vars_min_depend.append(depend_submatrix.columns[[var_a_idx, var_b_idx]].tolist())
        _min_depend.append(_min)
    return _vars_min_depend, _min_depend