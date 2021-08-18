import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

from dalex.model_explanations._variable_importance.object import VariableImportance
from dalex.model_explanations._variable_importance import utils
from ... import _theme, _global_checks
from .._predict_aspect_importance.utils import calculate_min_depend
from . import plot, checks


class ModelAspectImportance(VariableImportance):
    """Calculate model-level aspect importance

    Parameters
    ----------
    variable_groups : dict of lists 
        Variables grouped in aspects to calculate their importance.
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
    depend_method: {'assoc', 'pps'} or function, optional
        The method of calculating the dependencies between variables (i.e. the dependency 
        matrix). Default is `'assoc'`, which means the use of statistical association 
        (correlation coefficient, Cramér's V and eta-quared); 
        `'pps'` stands for Power Predictive Score.
        NOTE: When a function is passed, it is called with the `data` and it 
        must return a symmetric dependency matrix (`pd.DataFrame` with variable names as 
        columns and rows).
    corr_method : {'spearman', 'pearson', 'kendall'}, optional
        The method of calculating correlation between numerical variables 
        (default is `'spearman'`).
        NOTE: Ignored if `depend_method` is not `'assoc'`.
    agg_method : {'max', 'min', 'avg'}, optional
        The method of aggregating the PPS values for pairs of variables 
        (default is `'max'`).
        NOTE: Ignored if `depend_method` is not `'pps'`. 
    processes : int, optional
        Number of parallel processes to use in calculations. Iterated over `B`
        (default is `1`, which means no parallel computation).
    random_state : int, optional
        Set seed for random number generator (default is random seed).

    Attributes
    -----------
    result : pd.DataFrame
        Main result attribute of an explanation.
    variable_groups : dict of lists 
        Variables grouped in aspects to calculate their importance. 
    loss_function : function
        Loss function used to assess the variable importance.
    type : {'variable_importance', 'ratio', 'difference'}
        Type of transformation that will be applied to dropout loss.
    N : int
        Number of observations that will be sampled from the `explainer.data` attribute before
        the calculation of aspect importance. 
    B : int
        Number of permutation rounds to perform on each variable.
    depend_method : {'assoc', 'pps'}
        The method of calculating the dependencies between variables.
    corr_method : {'spearman', 'pearson', 'kendall'}
        The method of calculating correlation between numerical variables.
    agg_method : {'max', 'min', 'avg'}
        The method of aggregating the PPS values for pairs of variables.
    processes : int
        Number of parallel processes to use in calculations. Iterated over `B`.
    random_state : int
        Set seed for random number generator.
    """
    def __init__(
        self,
        variable_groups,
        loss_function="rmse",
        type="variable_importance",
        N=1000,
        B=10,
        depend_method="assoc",
        corr_method="spearman",
        agg_method="max",
        processes=1,
        random_state=None,
        _min_depend=None,
        _is_aspect_model_parts=True,
    ):
        super().__init__(
            loss_function,
            type,
            N,
            B,
            None,
            variable_groups,
            True,
            processes,
            random_state,
        )
        _depend_method, _corr_method, _agg_method = checks.check_method_depend(depend_method, corr_method, agg_method)
        self.result = pd.DataFrame()
        self._min_depend = _min_depend
        self.depend_method = _depend_method
        self.corr_method = _corr_method
        self.agg_method = _agg_method
        self._is_aspect_model_parts = _is_aspect_model_parts

    def fit(self, explainer):
        """Calculate the result of explanation
        Fit method makes calculations in place and changes the attributes.

        Parameters
        ----------
        explainer : Explainer object
            Model wrapper created using the Explainer class.
        
        Returns
        -----------
        None
        """
        super().fit(explainer)

        self.result["variables_names"] = self.result["variable"].map(
            self.variable_groups
        )
        baseline = self.result[self.result["variable"] == "_full_model_"][
            "dropout_loss"
        ].values[0]
        self.result = self.result.assign(
            dropout_loss_change=lambda x: x["dropout_loss"] - baseline
        )
        self.result = self.result.rename(columns={"variable": "aspect_name"})
        if self._is_aspect_model_parts:
            self.result.insert(4, "min_depend", None)
            self.result.insert(5, "vars_min_depend", None)
            if self._min_depend is not None:
                for index, row in self.result[1:-1].iterrows():
                    _matching_row = self._min_depend.loc[
                        self._min_depend.variables == set(row.variables_names)
                    ]
                    min_dep = _matching_row.min_depend.values[0]
                    vars_min_depend = _matching_row.vars_min_depend.values[0]
                    self.result.at[index, "min_depend"] = min_dep
                    self.result.at[index, "vars_min_depend"] = vars_min_depend
            else:
                vars_min_depend, min_depend = calculate_min_depend(
                    self.result[1:-1].variables_names,
                    explainer.data,
                    self.depend_method,
                    self.corr_method,
                    self.agg_method,
                )
                min_depend.append(None)
                min_depend.insert(0, None)
                vars_min_depend.append(None)
                vars_min_depend.insert(0, None)
                self.result["min_depend"] = min_depend
                self.result["vars_min_depend"] = vars_min_depend

            self.result = self.result[
                [
                    "aspect_name",
                    "variables_names",
                    "dropout_loss",
                    "dropout_loss_change",
                    "min_depend",
                    "vars_min_depend",
                    "label",
                ]
            ]
        else:
            self.result = self.result[
                [
                    "aspect_name",
                    "variables_names",
                    "dropout_loss",
                    "dropout_loss_change",
                    "label",
                ]
            ]

    def plot(
        self,
        objects=None,
        max_aspects=10,
        show_variables_names=True,
        digits=3,
        rounding_function=np.around,
        bar_width=25,
        split=("model", "aspect"),
        title="Model Aspect Importance",
        vertical_spacing=None,
        show=True,
    ):
        """Plot the Model Aspect Importance explanation.

        Parameters
        -----------
        objects : ModelAspectImportance object or array_like of ModelAspectImportance objects
            Additional objects to plot in subplots (default is `None`).
        max_aspects : int, optional
            Maximum number of aspects that will be presented for for each subplot
            (default is `10`).
        show_variables_names : bool, optional
            `True` shows names of variables grouped in aspects; `False` shows names of aspects
            (default is `True`).
        digits : int, optional
            Number of decimal places (`np.around`) to round contributions.
            See `rounding_function` parameter (default is `3`).
        rounding_function : function, optional
            A function that will be used for rounding numbers (default is `np.around`).
        bar_width : float, optional
            Width of bars in px (default is `25`).
        split : {'model', 'aspect'}, optional
            Split the subplots by model or aspect (default is `'model'`).
        title : str, optional
            Title of the plot (default is `"Model Aspect Importance"`).
        vertical_spacing : float <0, 1>, optional
            Ratio of vertical space between the plots (default is `0.2/number of rows`).
        show : bool, optional
            `True` shows the plot; `False` returns the plotly Figure object that can
            be edited or saved using the `write_image()` method (default is `True`).

        Returns
        -----------
        None or plotly.graph_objects.Figure
            Return figure that can be edited or saved. See `show` parameter.
        """

        if isinstance(split, tuple):
            split = split[0]

        if split not in ("model", "aspect"):
            raise TypeError("split should be 'model' or 'aspect'")

        if not self._is_aspect_model_parts:
            objects = None

        # are there any other objects to plot?
        if objects is None:
            n = 1
            _result_df = self.result.copy()
            if split == "aspect":  # force split by model if only one explainer
                split = "model"
        elif isinstance(
            objects, self.__class__
        ):  # allow for objects to be a single element
            n = 2
            _result_df = pd.concat([self.result.copy(), objects.result.copy()])
        elif isinstance(objects, (list, tuple)):  # objects as tuple or array
            n = len(objects) + 1
            _result_df = self.result.copy()
            for ob in objects:
                _global_checks.global_check_object_class(ob, self.__class__)
                _result_df = pd.concat([_result_df, ob.result.copy()])
        else:
            _global_checks.global_raise_objects_class(objects, self.__class__)

        dl = _result_df.loc[
            _result_df.aspect_name != "_baseline_", "dropout_loss"
        ].to_numpy()
        min_max_margin = dl.ptp() * 0.15
        min_max = [dl.min() - min_max_margin, dl.max() + min_max_margin]

        # take out full model
        best_fits = _result_df[_result_df.aspect_name == "_full_model_"]

        # this produces dropout_loss_x and dropout_loss_y columns
        _result_df = _result_df.merge(
            best_fits[["label", "dropout_loss"]], how="left", on="label"
        )
        # remove full_model and baseline
        _result_df = _result_df[
            (_result_df.aspect_name != "_full_model_")
            & (_result_df.aspect_name != "_baseline_")
        ]
        if self._is_aspect_model_parts:
            _result_df = _result_df[
                [
                    "label",
                    "aspect_name",
                    "dropout_loss_x",
                    "dropout_loss_y",
                    "variables_names",
                    "min_depend",
                    "vars_min_depend",
                ]
            ].rename(
                columns={
                    "dropout_loss_x": "dropout_loss",
                    "dropout_loss_y": "full_model",
                }
            )
            # calculate order of bars or variable plots (split = 'aspect')
            # get variable permutation
            perm = (
                _result_df[["aspect_name", "dropout_loss"]]
                .groupby("aspect_name")
                .mean()
                .reset_index()
                .sort_values("dropout_loss", ascending=False)
                .aspect_name.values
            )
            model_names = _result_df["label"].unique().tolist()

            if len(model_names) != n:
                raise ValueError("label must be unique for each model")
        else:
            _result_df = _result_df[
                [
                    "label",
                    "aspect_name",
                    "dropout_loss_x",
                    "dropout_loss_y",
                ]
            ].rename(
                columns={
                    "dropout_loss_x": "dropout_loss",
                    "dropout_loss_y": "full_model",
                }
            )
            model_names=None

        plot_height = 78 + 71

        colors = _theme.get_default_colors(n, "bar")

        if vertical_spacing is None:
            vertical_spacing = 0.2 / n

        x_title = "drop-out loss" if self._is_aspect_model_parts else None
        # init plot
        fig = make_subplots(
            rows=n,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=vertical_spacing,
            x_title=x_title,
            subplot_titles=model_names,
        )
        if split == "model":
            # split df by model
            df_list = [v for k, v in _result_df.groupby("label", sort=False)]

            for i, df in enumerate(df_list):
                m = df.shape[0]
                if max_aspects is not None and max_aspects < m:
                    m = max_aspects

                # take only m variables (for max_aspects)
                # sort rows of df by variable permutation and drop unused variables
                if self._is_aspect_model_parts:
                    df = (
                        df.sort_values("dropout_loss")
                        .tail(m)
                        .set_index("aspect_name")
                        .reindex(perm)
                        .dropna()
                        .reset_index()
                    )

                baseline = df.iloc[0, df.columns.get_loc("full_model")]

                df = df.assign(difference=lambda x: x["dropout_loss"] - baseline)

                lt = df.difference.apply(
                    lambda val: "+" + str(rounding_function(np.abs(val), digits))
                    if val > 0
                    else str(rounding_function(np.abs(val), digits))
                )
                if self._is_aspect_model_parts:
                    tt = df.apply(
                        lambda row: plot.tooltip_text_aspect(
                            row, rounding_function, digits
                        ),
                        axis=1,
                    )
                else:
                    tt = df.apply(
                        lambda row: plot.tooltip_text_single(
                            row, rounding_function, digits
                        ),
                        axis=1,
                    )
                df = df.assign(label_text=lt, tooltip_text=tt)

                fig.add_shape(
                    type="line",
                    x0=baseline,
                    x1=baseline,
                    y0=-1,
                    y1=m,
                    yref="paper",
                    xref="x",
                    line={"color": "#371ea3", "width": 1.5, "dash": "dot"},
                    row=i + 1,
                    col=1,
                )

                if show_variables_names:
                    y_axis_ticks = [
                        ", ".join(variables_list)
                        for variables_list in df["variables_names"]
                    ]
                else:
                    y_axis_ticks = df["aspect_name"]

                fig.add_bar(
                    orientation="h",
                    y=y_axis_ticks,
                    x=df["difference"].tolist(),
                    textposition="outside",
                    text=df["label_text"].tolist(),
                    marker_color=colors[i],
                    base=baseline,
                    hovertext=df["tooltip_text"].tolist(),
                    hoverinfo="text",
                    hoverlabel={"bgcolor": "rgba(0,0,0,0.8)"},
                    showlegend=False,
                    row=i + 1,
                    col=1,
                )

                fig.update_yaxes(
                    {
                        "type": "category",
                        "autorange": "reversed",
                        "gridwidth": 2,
                        "automargin": True,
                        "ticks": "outside",
                        "tickcolor": "white",
                        "ticklen": 10,
                        "fixedrange": True,
                    },
                    row=i + 1,
                    col=1,
                )

                fig.update_xaxes(
                    {
                        "type": "linear",
                        "gridwidth": 2,
                        "zeroline": False,
                        "automargin": True,
                        "ticks": "outside",
                        "tickcolor": "white",
                        "ticklen": 3,
                        "fixedrange": True,
                    },
                    row=i + 1,
                    col=1,
                )

                plot_height += m * bar_width + (m + 1) * bar_width / 4 + 30
        elif split == "aspect":
            # split df by aspect
            df_list = [v for k, v in _result_df.groupby("aspect_name", sort=False)]

            n = len(df_list)
            if max_aspects is not None and max_aspects < n:
                n = max_aspects

            if vertical_spacing is None:
                vertical_spacing = 0.2 / n
            # init plot
            variable_names = perm[0:n]
            fig = make_subplots(
                rows=n,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=vertical_spacing,
                x_title=x_title,
                subplot_titles=variable_names,
            )

            df_dict = {e.aspect_name.array[0]: e for e in df_list}

            # take only n=max_aspects elements from df_dict
            for i in range(n):
                df = df_dict[perm[i]]
                m = df.shape[0]

                baseline = 0

                df = df.assign(difference=lambda x: x["dropout_loss"] - x["full_model"])

                lt = df.difference.apply(
                    lambda val: "+" + str(rounding_function(np.abs(val), digits))
                    if val > 0
                    else str(rounding_function(np.abs(val), digits))
                )
                tt = df.apply(
                    lambda row: plot.tooltip_text_aspect(row, rounding_function, digits),
                    axis=1,
                )
                df = df.assign(label_text=lt, tooltip_text=tt)

                fig.add_shape(
                    type="line",
                    x0=baseline,
                    x1=baseline,
                    y0=-1,
                    y1=m,
                    yref="paper",
                    xref="x",
                    line={"color": "#371ea3", "width": 1.5, "dash": "dot"},
                    row=i + 1,
                    col=1,
                )

                fig.add_bar(
                    orientation="h",
                    y=df["label"].tolist(),
                    x=df["dropout_loss"].tolist(),
                    textposition="outside",
                    text=df["label_text"].tolist(),
                    marker_color=colors,
                    base=baseline,
                    hovertext=df["tooltip_text"].tolist(),
                    hoverinfo="text",
                    hoverlabel={"bgcolor": "rgba(0,0,0,0.8)"},
                    showlegend=False,
                    row=i + 1,
                    col=1,
                )

                fig.update_yaxes(
                    {
                        "type": "category",
                        "autorange": "reversed",
                        "gridwidth": 2,
                        "automargin": True,
                        "ticks": "outside",
                        "tickcolor": "white",
                        "ticklen": 10,
                        "dtick": 1,
                        "fixedrange": True,
                    },
                    row=i + 1,
                    col=1,
                )

                fig.update_xaxes(
                    {
                        "type": "linear",
                        "gridwidth": 2,
                        "zeroline": False,
                        "automargin": True,
                        "ticks": "outside",
                        "tickcolor": "white",
                        "ticklen": 3,
                        "fixedrange": True,
                    },
                    row=i + 1,
                    col=1,
                )

                plot_height += m * bar_width + (m + 1) * bar_width / 4

        plot_height += (n - 1) * 70

        fig.update_xaxes({"range": min_max})
        fig.update_layout(
            title_text=title,
            title_x=0.15,
            font={"color": "#371ea3"},
            template="none",
            height=plot_height,
            margin={"t": 78, "b": 71, "r": 30},
        )

        if show:
            fig.show(config=_theme.get_default_config())
        else:
            return fig
