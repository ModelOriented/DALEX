import numpy as np
import pandas as pd
import warnings

from . import checks
from .. import _global_checks

def calculate_cat_assoc_matrix(data, categorical_variables, n):
    from scipy.stats import chi2_contingency
    cat_assoc_matrix = pd.DataFrame(
        index=categorical_variables, columns=categorical_variables, dtype=float
    )
    for i, variable_name_a in enumerate(categorical_variables):
        for j, variable_name_b in enumerate(categorical_variables):
            if i > j:
                continue   # matrix is symmetric
            elif i == j:
                cramers_V = 1
            else:
                # calculate Cramér’s V with bias correction
                cont_tab = pd.crosstab(data[variable_name_a], data[variable_name_b])
                r = cont_tab.shape[0]
                c = cont_tab.shape[1]
                chi2_test = chi2_contingency(cont_tab)
                phi = max(0, (chi2_test[0] / n) - (((r - 1) * (c - 1)) / (n - 1)))
                r = r - ((r - 1) ** 2 / (n - 1))
                c = c - ((c - 1) ** 2 / (n - 1))
                cramers_V = np.sqrt(phi / (min(c, r) - 1))
                cramers_V = checks.check_assoc_value(cramers_V)
            cat_assoc_matrix.loc[variable_name_a, variable_name_b] = cramers_V
            cat_assoc_matrix.loc[variable_name_b, variable_name_a] = cramers_V
    return cat_assoc_matrix


def calculate_cat_num_assoc_matrix(data, categorical_variables, numerical_variables, n):
    from scipy.stats import kruskal
    cat_num_assoc_matrix = pd.DataFrame(
        index=categorical_variables, columns=numerical_variables, dtype=float
    )
    for cat_variable_name in categorical_variables:
        for num_variable_name in numerical_variables:
            # calculate eta-squared
            samples = [
                gr_data[num_variable_name].values
                for gr_name, gr_data in data.groupby(cat_variable_name, as_index=False)
            ]
            kruskal_wallis_test = kruskal(*samples, nan_policy="omit")
            n_samples = len(samples)
            eta_squared = (kruskal_wallis_test.statistic - n_samples + 1) / (n - n_samples)
            eta_squared = checks.check_assoc_value(eta_squared)
            cat_num_assoc_matrix.loc[cat_variable_name, num_variable_name] = eta_squared
    return cat_num_assoc_matrix


def calculate_assoc_matrix(data, corr_method):
    corr_matrix = abs(data.corr(corr_method, numeric_only=True)) # get submatrix for only numerical variables
    numerical_variables = corr_matrix.columns 
    categorical_variables = list(set(data.columns) - set(numerical_variables))
    n = len(data)
    cat_assoc_matrix = calculate_cat_assoc_matrix(data, categorical_variables, n) # get submatrix for only categorical variables
    cat_num_assoc_matrix = calculate_cat_num_assoc_matrix(          # get submatrix for mixed variables
        data, categorical_variables, numerical_variables, n
    )
    tmp_matrix = pd.concat([corr_matrix, cat_num_assoc_matrix.T], axis=1)
    tmp_matrix_2 = pd.concat([cat_num_assoc_matrix, cat_assoc_matrix], axis=1)
    assoc_matrix = pd.concat([tmp_matrix, tmp_matrix_2])
    return assoc_matrix


def calculate_pps_matrix(data, agg_method):
    _global_checks.global_check_import('ppscore', 'depend_matrix with PPS calculation')
    import ppscore as pps
    pps_result = pps.matrix(data, sample=None)
    pps_matrix = pps_result[["x", "y", "ppscore"]].pivot(
        columns="x", index="y", values="ppscore"
    )
    pps_matrix.rename_axis(None, axis=1, inplace=True)
    pps_matrix.rename_axis(None, axis=0, inplace=True)
    # aggregate values to make symmetric matrix
    if agg_method == "max":
        pps_matrix = np.maximum(pps_matrix, pps_matrix.transpose())
    if agg_method == "min":
        pps_matrix = np.minimum(pps_matrix, pps_matrix.transpose())
    if agg_method == "mean":
        pps_matrix = (pps_matrix + pps_matrix.transpose()) / 2
    return pps_matrix


def calculate_depend_matrix(
    data, depend_method, corr_method, agg_method
):
    depend_matrix = pd.DataFrame()
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
    
    # if there is a non-varying column in data, there will be NaN values in the 'depend_matrix'.
    # replace NaN values on the diagonal with 1 and others with 0. 
    if depend_matrix.isnull().any(axis=None):
        warnings.warn("There were NaNs in `depend_matrix`. This is possibly because there is a feature in the data with only one unique value. Replacing NaN values on the diagonal with 1 and others with 0.")
        depend_matrix[depend_matrix.isnull()] = 0
        for i in range(depend_matrix.shape[0]):
            depend_matrix.iloc[i,i] = 1
    
    return depend_matrix


def calculate_linkage_matrix(depend_matrix, clust_method):
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    dissimilarity = 1 - abs(depend_matrix)
    ## https://www.kaggle.com/sgalella/correlation-heatmaps-with-hierarchical-clustering?scriptVersionId=24045077&cellId=10
    linkage_matrix = linkage(squareform(dissimilarity), clust_method)

    return linkage_matrix


def get_dendrogram_aspects_ordered(hierarchical_clustering_dendrogram, depend_matrix):
    # names of variables on axis
    tick_dict = dict(
        zip(
            hierarchical_clustering_dendrogram.layout.yaxis.tickvals,
            [[var] for var in hierarchical_clustering_dendrogram.layout.yaxis.ticktext]
        )
    )
    d = {k:v for v,k in enumerate(depend_matrix.columns)} # original order of columns
    
    # get names of variables for dendrogram traces
    _aspects_dendrogram_order = []
    for scatter in hierarchical_clustering_dendrogram.data:
        vars_list = tick_dict[scatter.y[1]] + tick_dict[scatter.y[2]]
        vars_list.sort(key=d.get)
        tick_dict[np.mean(scatter.y[1:3])] = vars_list
        _aspects_dendrogram_order.append(
            vars_list
        )
    
    # get min dependency between grouped variables
    _vars_min_depend, _min_depend = get_min_depend_from_matrix(depend_matrix, _aspects_dendrogram_order)

    return pd.DataFrame({
        "variable_names": _aspects_dendrogram_order,
        "min_depend": _min_depend,
        "vars_min_depend": _vars_min_depend
    })


def get_min_depend_from_matrix(depend_matrix, variables_list): 
    _vars_min_depend = []
    _min_depend = []
    for vars in variables_list:
        if isinstance(vars, (list, tuple, np.ndarray)):
            depend_submatrix = abs(depend_matrix).loc[list(vars), list(vars)]
            _min = np.array(depend_submatrix).min()
            _min_idx = np.where(depend_submatrix == _min)
            var_a_idx, var_b_idx = _min_idx[0][0], _min_idx[1][0]
            _vars_min_depend.append(np.array(depend_submatrix.columns[[var_a_idx, var_b_idx]]))
            _min_depend.append(_min)
        else:
            _vars_min_depend.append(None)
            _min_depend.append(None)
    return _vars_min_depend, _min_depend


def calculate_min_depend(
    variables_list,
    data,
    depend_method="assoc",
    corr_method="spearman",
    agg_method="max",
): 
    depend_matrix = calculate_depend_matrix(data, depend_method, corr_method, agg_method)
    return get_min_depend_from_matrix(depend_matrix, variables_list)
