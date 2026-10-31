import numpy as np
import pandas as pd

from dalex.aspect._model_aspect_importance.object import ModelAspectImportance
from dalex.aspect._predict_aspect_importance.object import PredictAspectImportance
from dalex.aspect._model_triplot.object import ModelTriplot
from dalex.aspect._predict_triplot.object import PredictTriplot

from . import utils, checks, plot
from .. import _theme
from .._explanation import ResultTables, CompatibilityTable
from ._native_dependence import matrix_values
from ._table import AspectTable
import narwhals.stable.v2 as nw


class Aspect(ResultTables):
    """Create Aspect

    Explanation methods that do not take into account dependencies between variables
    can produce misleading results. This class creates a representation of a model based
    on an Explainer object. In addition, it calculates the relationships between
    the variables that can be used to create explanations. Methods of this class produce
    explanation objects, that contain the main result attribute, and can be visualised
    using the plot method.

    The `explainer` is the only required parameter.

    Parameters
    ----------
    explainer : Explainer object
        Model wrapper created using the Explainer class.
    depend_method: {'assoc', 'pps'} or function, optional
        The method of calculating the dependencies between variables (i.e. the dependency
        matrix). Default is `'assoc'`, which means the use of statistical association
        (correlation coefficient, Cramér's V based on Pearson's chi-squared statistic 
        and eta-quared based on Kruskal-Wallis H-statistic);
        `'pps'` stands for Power Predictive Score.
        NOTE: When a function is passed, it is called with the `explainer.data` and it
        must return a symmetric dependency matrix (`pd.DataFrame` with variable names as
        columns and rows).
    clust_method : {'complete', 'single', 'average', 'weighted', 'centroid', 'median', 'ward'}, optional
        The linkage algorithm to use for variables hierarchical clustering
        (default is `'complete'`).
    corr_method : {'spearman', 'pearson', 'kendall'}, optional
        The method of calculating correlation between numerical variables
        (default is `'spearman'`).
        NOTE: Ignored if `depend_method` is not `'assoc'`.
    agg_method : {'max', 'min', 'avg'}, optional
        The method of aggregating the PPS values for pairs of variables
        (default is `'max'`).
        NOTE: Ignored if `depend_method` is not `'pps'`.

    Attributes
    --------
    explainer : Explainer object
        Model wrapper created using the Explainer class.
    depend_method : {'assoc', 'pps'} or function
        The method of calculating the dependencies between variables.
    clust_method : {'complete', 'single', 'average', 'weighted', 'centroid', 'median', 'ward'}
        The linkage algorithm to use for variables hierarchical clustering.
    corr_method : {'spearman', 'pearson', 'kendall'}
        The method of calculating correlation between numerical variables.
    agg_method : {'max', 'min', 'avg'}
        The method of aggregating the PPS values for pairs of variables.
    depend_matrix : pd.DataFrame
        The dependency matrix (with variable names as columns and rows).
    linkage_matrix :
        The hierarchical clustering of variables encoded as a `scipy` linkage matrix.

    Notes
    -----
    - assoc, eta-squared: http://tss.awf.poznan.pl/files/3_Trends_Vol21_2014__no1_20.pdf
    - assoc, Cramér's V: http://stats.lse.ac.uk/bergsma/pdf/cramerV3.pdf
    - PPS: https://github.com/8080labs/ppscore
    - triplot: https://arxiv.org/abs/2104.03403
    """

    depend_matrix = CompatibilityTable()
    _dendrogram_aspects_ordered = CompatibilityTable()
    _full_hierarchical_aspect_importance = CompatibilityTable()

    def _native_depend_matrix(self):
        return self.__dict__.get('_compatibility_depend_matrix', self.__dict__['depend_matrix'])

    @staticmethod
    def _set_result_label(explanation, label):
        table = explanation.__dict__.get('result')
        if isinstance(table, AspectTable) and '_compatibility_result' not in explanation.__dict__:
            table.data = nw.from_native(table.data, eager_only=True).with_columns(nw.lit(label).alias('label')).to_native()
        else:
            explanation.result['label'] = label

    def __init__(
        self,
        explainer,
        depend_method="assoc",
        clust_method="complete",
        corr_method="spearman",
        agg_method="max",
    ):  
        _depend_method, _corr_method, _agg_method = checks.check_method_depend(depend_method, corr_method, agg_method)
        self.explainer = explainer
        self.depend_method = _depend_method
        self.clust_method = clust_method
        self.corr_method = _corr_method
        self.agg_method = _agg_method
        self.depend_matrix = utils.calculate_depend_matrix(
            self.explainer.data, self.depend_method, self.corr_method, self.agg_method, native=True
        )
        self.linkage_matrix = utils.calculate_linkage_matrix(
            self._native_depend_matrix(), clust_method
        )
        self._hierarchical_clustering_dendrogram = plot.plot_dendrogram(
            self.linkage_matrix, matrix_values(self._native_depend_matrix())[1]
        )
        self._dendrogram_aspects_ordered = utils.get_dendrogram_aspects_ordered(
            self._hierarchical_clustering_dendrogram, self._native_depend_matrix(), native=True
        )
        self._full_hierarchical_aspect_importance = None
        self._mt_params = None

    def get_aspects(self, h=0.5, n=None):
        from scipy.cluster.hierarchy import fcluster
        """Form aspects of variables from the hierarchical clustering

        Parameters
        ----------
        h : float, optional
            Threshold to apply when forming aspects, i.e., the minimum value of the dependency
            between the variables grouped in one aspect (default is `0.5`).
            NOTE: Ignored if `n` is not `None`.
        n : int, optional
            Maximum number of aspects to form 
            (default is `None`, which means the use of `h` parameter).

        Returns
        -------
        dict of lists
            Variables grouped in aspects, e.g. `{'aspect_1': ['x1', 'x2'], 'aspect_2': ['y1', 'y2']}`.
        """
        if n is None:
            aspect_label = fcluster(self.linkage_matrix, 1 - h, criterion="distance")
        else:
            aspect_label = fcluster(self.linkage_matrix, n, criterion="maxclust")
        names = matrix_values(self._native_depend_matrix())[1]
        groups = {}
        for name, label in zip(names, aspect_label):
            groups.setdefault(label, []).append(name)
        aspects_dict = {}
        i = 1
        for label in sorted(groups):
            features = groups[label]
            key = f"aspect_{i}" if len(features) > 1 else features[0]
            aspects_dict[key] = features
            i += len(features) > 1

        return aspects_dict

    def plot_dendrogram(
        self,
        title="Hierarchical clustering dendrogram",
        lines_interspace=20,
        rounding_function=np.round,
        digits=3,
        show=True,
    ):
        """Plot the hierarchical clustering dendrogram of variables

        Parameters
        ----------
        title : str, optional
            Title of the plot (default is "Hierarchical clustering dendrogram").
        lines_interspace : float, optional
            Interspace between lines of dendrogram in px (default is `20`).
        rounding_function : function, optional
            A function that will be used for rounding numbers (default is `np.around`).
        digits : int, optional
            Number of decimal places (`np.around`) to round contributions.
            See `rounding_function` parameter (default is `3`).
        show : bool, optional
            `True` shows the plot; `False` returns the plotly Figure object that can
            be edited or saved using the `write_image()` method (default is `True`).

        Returns
        -------
        None or plotly.graph_objects.Figure
            Return figure that can be edited or saved. See `show` parameter.
        """
        m = len(matrix_values(self._native_depend_matrix())[1])
        plot_height = 78 + 71 + m * lines_interspace + (m + 1) * lines_interspace / 4
        fig = self._hierarchical_clustering_dendrogram
        # Read native storage directly. The descriptor is the writable legacy
        # pandas boundary, so accessing it here would publish a view just to
        # prepare tooltips. An explicitly read/edited view stays authoritative.
        aspects = self.__dict__.get('_compatibility__dendrogram_aspects_ordered',
                                    self.__dict__['_dendrogram_aspects_ordered'])
        fig = plot.add_text_and_tooltips_to_dendrogram(
            fig, aspects, rounding_function, digits
        )
        fig = plot._add_points_on_dendrogram_traces(fig)
        fig.update_layout(
            title={"text": title, "x": 0.15},
            yaxis={"automargin": True, "autorange": "reversed"},
            height=plot_height,
        )
        if show:
            fig.show(config=_theme.get_default_config())
        else:
            return fig

    def predict_parts(
        self,
        new_observation,
        variable_groups=None,
        type="default",
        h=0.5,
        N=2000,
        B=25,
        n_aspects=None,
        sample_method="default",
        f=2,
        label=None,
        processes=1,
        random_state=None,
    ):
        """Calculate predict-level aspect importance

        Parameters
        ----------
        new_observation : pd.Series or np.ndarray (1d) or pd.DataFrame (1,p)
            An observation for which a prediction needs to be explained.
        variable_groups : dict of lists or None
            Variables grouped in aspects to calculate their importance (default is `None`).
        type : {'default', 'shap'}, optional
            Type of aspect importance/attributions (default is `'default'`, which means
            the use of simplified LIME method).
        h : float, optional
            Threshold to apply when forming aspects, i.e., the minimum value of the dependency
            between the variables grouped in one aspect (default is `0.5`).
        N : int, optional
            Number of observations that will be sampled from the `explainer.data` attribute
            before the calculation of aspect importance (default is `2000`).
        B : int, optional
            Parameter specific for `type == 'shap'`. Number of random paths to calculate aspect
            attributions (default is `25`).
            NOTE: Ignored if `type` is not `'shap'`.
        n_aspects : int, optional
            Parameter specific for `type == 'default'`. Maximum number of non-zero importances, i.e.
            coefficients after lasso fitting (default is `None`, which means the linear regression is used).
            NOTE: Ignored if `type` is not `'default'`.
        sample_method : {'default', 'binom'}, optional
            Parameter specific for `type == 'default'`. Sampling method for creating binary matrix
            used as mask for replacing aspects in sampled data (default is `'default'`, which means
            it randomly replaces one or two zeros per row; `'binom'` replaces random number of zeros
            per row).
            NOTE: Ignored if `type` is not `'default'`.
        f : int, optional
            Parameter specific for `type == 'default'` and `sample_method == 'binom'`. Parameter
            controlling average number of replaced zeros for binomial sampling (default is `2`).
            NOTE: Ignored if `type` is not `'default'` or `sample_method` is not `'binom'`.
        label : str, optional
            Name to appear in result and plots. Overrides default.
        processes : int, optional
            Parameter specific for `type == 'shap'`. Number of parallel processes to use in calculations.
            Iterated over `B` (default is `1`, which means no parallel computation).
        random_state : int, optional
            Set seed for random number generator (default is random seed).

        Returns
        -------
        PredictAspectImportance class object
            Explanation object containing the main result attribute and the plot method.
        """

        if variable_groups is None:
            variable_groups = self.get_aspects(h)

        pai = PredictAspectImportance(
            variable_groups,
            type,
            N,
            B,
            n_aspects,
            sample_method,
            f,
            self.depend_method,
            self.corr_method,
            self.agg_method,
            processes,
            random_state,
            _depend_matrix=self._native_depend_matrix()
        )

        pai.fit(self.explainer, new_observation)

        if label is not None:
            self._set_result_label(pai, label)

        return pai

    def model_parts(
        self,
        variable_groups=None,
        h=0.5,
        loss_function=None,
        type="variable_importance",
        N=1000,
        B=10,
        processes=1,
        label=None,
        random_state=None,
    ):
        """Calculate model-level aspect importance

        Parameters
        ----------
        variable_groups : dict of lists or None
            Variables grouped in aspects to calculate their importance (default is `None`).
        h : float, optional
            Threshold to apply when forming aspects, i.e., the minimum value of the dependency
            between the variables grouped in one aspect (default is `0.5`).
        loss_function :  {'rmse', '1-auc', 'mse', 'mae', 'mad'} or function, optional
            If string, then such loss function will be used to assess aspect importance
            (default is `'rmse'` or `'1-auc'`, depends on `explainer.model_type` attribute).
        type : {'variable_importance', 'ratio', 'difference'}, optional
            Type of transformation that will be applied to dropout loss
            (default is `'variable_importance'`, which is Permutational Variable Importance).
        N : int, optional
            Number of observations that will be sampled from the `explainer.data` attribute before
            the calculation of aspect importance. `None` means all `data` (default is `1000`).
        B : int, optional
            Number of permutation rounds to perform on each variable (default is `10`).
        processes : int, optional
            Number of parallel processes to use in calculations. Iterated over `B`
            (default is `1`, which means no parallel computation).
        label : str, optional
            Name to appear in result and plots. Overrides default.
        random_state : int, optional
            Set seed for random number generator (default is random seed).

        Returns
        -------
        ModelAspectImportance class object
            Explanation object containing the main result attribute and the plot method.
        """

        loss_function = checks.check_method_loss_function(self.explainer, loss_function)
        mai_result = None

        if variable_groups is None:
            variable_groups = self.get_aspects(h)

            # get results from triplot if it was precalculated with the same params
            full = self.__dict__.get('_compatibility__full_hierarchical_aspect_importance',
                                     self.__dict__.get('_full_hierarchical_aspect_importance'))
            if full is not None:
                if (
                    self._mt_params["loss_function"] == loss_function
                    and self._mt_params["N"] == N
                    and self._mt_params["B"] == B
                    and self._mt_params["type"] == type
                    and self._mt_params.get('random_state') == random_state
                    and self._mt_params.get('processes') == processes
                ):
                    h = min(1, h)
                    if isinstance(full, AspectTable):
                        heights = nw.from_native(full.data, eager_only=True)['h'].to_numpy()
                        selected = np.unique(heights[heights >= h])[0]
                        ids = np.flatnonzero(heights == selected).tolist()
                        columns = [name for name in full._columns() if name != 'h']
                        mai_result = full.subset(ids, columns=columns, keep_index=True)
                    else:
                        selected = np.unique(full.loc[full.h >= h].h)[0]
                        mai_result = full.loc[full.h == selected]


        ai = ModelAspectImportance(
            loss_function=loss_function,
            type=type,
            N=N,
            B=B,
            variable_groups=variable_groups,
            processes=processes,
            random_state=random_state,
            _depend_matrix=self._native_depend_matrix()
        )

        # calculate if there was no results
        if mai_result is None:
            ai.fit(self.explainer)
        else: 
            if not isinstance(mai_result, AspectTable):
                mai_result = mai_result[
                    ["aspect_name", "variable_names", "dropout_loss", "dropout_loss_change",
                     "min_depend", "vars_min_depend", "label"]
                ]
            ai.result = mai_result

        if label is not None:
            self._set_result_label(ai, label)

        return ai

    def predict_triplot(
        self,
        new_observation,
        type="default",
        N=2000,
        B=25,
        sample_method="default",
        f=2,
        processes=1,
        random_state=None,
    ):
        """Calculate predict-level hierarchical aspect importance

        Parameters
        ----------
        new_observation : pd.Series or np.ndarray (1d) or pd.DataFrame (1,p)
            An observation for which a prediction needs to be explained.
        type : {'default', 'shap'}, optional
            Type of aspect importance/attributions (default is `'default'`, which means
            the use of simplified LIME method).
        N : int, optional
            Number of observations that will be sampled from the `explainer.data` attribute
            before the calculation of aspect importance (default is `2000`).
        B : int, optional
            Parameter specific for `type == 'shap'`. Number of random paths to calculate aspect
            attributions (default is `25`).
            NOTE: Ignored if `type` is not `'shap'`.
        sample_method : {'default', 'binom'}, optional
            Parameter specific for `type == 'default'`. Sampling method for creating binary matrix
            used as mask for replacing aspects in data (default is `'default'`, which means
            it randomly replaces one or two zeros per row; `'binom'` replaces random number of zeros
            per row).
            NOTE: Ignored if `type` is not `'default'`.
        f : int, optional
            Parameter specific for `type == 'default'` and `sample_method == 'binom'`. Parameter
            controlling average number of replaced zeros for binomial sampling (default is `2`).
            NOTE: Ignored if `type` is not `'default'` or `sample_method` is not `'binom'`.
        processes : int, optional
            Number of parallel processes to use in calculations. Iterated over `B`
            (default is `1`, which means no parallel computation).
        random_state : int, optional
            Set seed for random number generator (default is random seed).

        Returns
        -------
        PredictTriplot class object
            Explanation object containing the main result attribute and the plot method.
        """

        pt = PredictTriplot(type, N, B, sample_method, f, processes, random_state)

        pt.fit(self, new_observation)

        return pt

    def model_triplot(
        self,
        loss_function=None,
        type="variable_importance",
        N=1000,
        B=10,
        processes=1,
        random_state=None,
    ):
        """Calculate model-level hierarchical aspect importance

        Parameters
        ----------
        loss_function :  {'rmse', '1-auc', 'mse', 'mae', 'mad'} or function, optional
            If string, then such loss function will be used to assess aspect importance
            (default is `'rmse'` or `'1-auc'`, depends on `explainer.model_type` attribute).
        type : {'variable_importance', 'ratio', 'difference'}, optional
            Type of transformation that will be applied to dropout loss
            (default is `'variable_importance'`, which is Permutational Variable Importance).
        N : int, optional
            Number of observations that will be sampled from the `explainer.data` attribute before
            the calculation of aspect importance. `None` means all `data` (default is `1000`).
        B : int, optional
            Number of permutation rounds to perform on each variable (default is `10`).
        processes : int, optional
            Number of parallel processes to use in calculations. Iterated over `B`
            (default is `1`, which means no parallel computation).
        random_state : int, optional
            Set seed for random number generator (default is random seed).

        Returns
        -------
        ModelTriplot class object
            Explanation object containing the main result attribute and the plot method.
        """

        
        loss_function = checks.check_method_loss_function(self.explainer, loss_function) # get proper loss_function for model_type
        mt = ModelTriplot(loss_function, type, N, B, processes, random_state)
        self._mt_params = {"loss_function": loss_function, "type": type, "N": N, "B": B,
                           "random_state": random_state, "processes": processes}
        mt.fit(self)

        return mt
