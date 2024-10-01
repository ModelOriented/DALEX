import numpy as np
import pandas as pd
import plotly.graph_objs as go
from copy import deepcopy

from dalex import _theme, _global_checks
from dalex._explanation import Explanation
from dalex.aspect._predict_aspect_importance.object import PredictAspectImportance

from . import checks, plot, utils



class PredictTriplot(Explanation):
    """Calculate predict-level hierarchical aspect importance

    Parameters
    ----------
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

    Attributes
    -----------
    result : pd.DataFrame
        Main result attribute of an explanation.
    single_variable_importance : pd.DataFrame
        Additional result attribute of an explanation (it contains information 
        about the importance of individual variables).
    prediction : float
        Prediction for `new_observation`.
    intercept : float
        Average prediction for `data`.
    type : {'default', 'shap'}
        Type of aspect importance/attributions to calculate.
    N : int
        Number of observations that will be sampled from the `data` attribute
        before the calculation of aspect importance.
    B : int
        Number of random paths to calculate aspect attributions.
    sample_method : {'default', 'binom'}
        Sampling method for creating binary matrix used as mask for replacing aspects in sampled data.
    f : int
        Average number of replaced zeros for binomial sampling.
    processes : int
        Number of parallel processes to use in calculations. Iterated over `B`.
    random_state : int or None
        Set seed for random number generator.

    Notes
    -----
    - https://arxiv.org/abs/2104.03403
    """
    def __init__(
        self,
        type="default",
        N=2000,
        B=25,
        sample_method="default",
        f=2,
        processes=1,
        random_state=None,
    ):
        types = ("default", "shap")
        aliases = {"simplified_lime": "default", "lime": "default"}
        _type = checks.check_method_type(type, types, aliases)
        self.type = _type
        _processes = checks.check_processes(processes)
        self.processes = _processes
        _random_state = checks.check_random_state(random_state)
        self.random_state = _random_state
        self.N = N
        self.B = checks.check_B(B)
        self.sample_method = sample_method
        self.f = f
        self.result = pd.DataFrame()
        self.single_variable_importance = None
        self._hierarchical_clustering_dendrogram = None

    def _repr_html_(self):
        return self.result._repr_html_()

    def fit(self, aspect, new_observation):
        """Calculate the result of explanation
        Fit method makes calculations in place and changes the attributes.

        Parameters
        ----------
        aspect : Aspect object
            Explainer wrapper created using the Aspect class.
        new_observation : pd.Series or np.ndarray (1d) or pd.DataFrame (1,p)
            An observation for which a prediction needs to be explained.
        
        Returns
        -----------
        None
        """

        _new_observation = checks.check_new_observation(new_observation, aspect.explainer)
        checks.check_columns_in_new_observation(_new_observation, aspect.explainer)

        self._hierarchical_clustering_dendrogram = aspect._hierarchical_clustering_dendrogram

        self.prediction = aspect.explainer.predict(_new_observation)[0]
        self.intercept = aspect.explainer.y_hat.mean()

        ## middle plot data
        self.result = (
            utils.calculate_predict_hierarchical_importance(
                aspect,
                _new_observation,
                self.type,
                self.N,
                self.B,
                self.sample_method,
                self.f,
                self.processes,
                self.random_state,
            )
        )

        self.result.insert(3, "min_depend", None)
        self.result.insert(4, "vars_min_depend", None)
        for index, row in self.result.iterrows():
            _matching_row = aspect._dendrogram_aspects_ordered.loc[
                pd.Series(map(set, aspect._dendrogram_aspects_ordered.variable_names))
                == set(row.variable_names)
            ]
            min_dep = _matching_row.min_depend.values[0]
            vars_min_depend = _matching_row.vars_min_depend.values[0]
            self.result.at[index, "min_depend"] = min_dep
            self.result.at[
                index, "vars_min_depend"
            ] = vars_min_depend

        ## left plot data
        self.single_variable_importance = utils.calculate_single_variable_importance(
            aspect,
            _new_observation,
            self.type,
            self.N,
            self.B, 
            self.sample_method,
            self.f,
            self.processes,
            self.random_state
        )
        
    def plot(
        self,
        absolute_value=False,
        digits=3,
        rounding_function=np.around,
        bar_width=25,
        width=1500,
        abbrev_labels=0,
        vcolors=None,
        title="Predict Triplot",
        widget=False,
        show=True
    ):
        """Plot the Predict Triplot explanation (triplot visualization).

        Parameters
        ----------
        absolute_value : bool, optional
            If `True` aspect importance values are drawn as absolute values 
            (default is `False`).
        digits : int, optional
            Number of decimal places (`np.around`) to round contributions.
            See `rounding_function` parameter (default is `3`).
        rounding_function : function, optional
            A function that will be used for rounding numbers (default is `np.around`).
        bar_width : float, optional
            Width of bars in px (default is `25`).
        width : float, optional
            Width of triplot in px (default is `1500`).
        abbrev_labels : int, optional
            If greater than 0, labels for axis Y will be abbreviated according to this parameter
            (default is `0`, which means no abbreviation).
        vcolors : 2-tuple of str, optional
            Color of bars (default is `["#8bdcbe", "#f05a71"]`).
        title : str, optional
            Title of the plot (default is `"Predict Triplot"`).
        widget : bool, optional
            If `True` triplot interactive widget version is generated
            (default is `False`).
        show : bool, optional
            `True` shows the plot; `False` returns the plotly Figure object 
            (default is `True`).
            NOTE: Ignored if `widget` is `True`.

        Returns
        -------
        None or plotly.graph_objects.Figure or ipywidgets.HBox with plotly.graph_objs._figurewidget.FigureWidget
            Return figure that can be edited or saved. See `show` parameter.
        """    
        _global_checks.global_check_import('kaleido', 'Predict Triplot')
        ## right plot
        hierarchical_clustering_dendrogram_plot_without_annotations = go.Figure(
            self._hierarchical_clustering_dendrogram
        )
        variables_order = list(
            hierarchical_clustering_dendrogram_plot_without_annotations.layout.yaxis.ticktext
        )

        ## middle plot
        (
            hierarchical_importance_dendrogram_plot_without_annotations,
            updated_dendro_traces,
        ) = plot.plot_predict_hierarchical_importance(
            hierarchical_clustering_dendrogram_plot_without_annotations,
            self.result,
            rounding_function,
            digits,
            absolute_value,
            self.type,
        )

        hierarchical_clustering_dendrogram_plot = plot.add_text_to_dendrogram(
            hierarchical_clustering_dendrogram_plot_without_annotations,
            updated_dendro_traces,
            rounding_function,
            digits,
            type="clustering",
        )

        hierarchical_importance_dendrogram_plot = plot.add_text_to_dendrogram(
            hierarchical_importance_dendrogram_plot_without_annotations,
            updated_dendro_traces,
            rounding_function,
            digits,
            type="importance",
        )

        ## left plot
        fig = plot.plot_single_aspects_importance(
            self.single_variable_importance,
            variables_order,
            rounding_function,
            digits,
            vcolors,
        )
        fig.layout["xaxis"]["range"] = (
            fig.layout["xaxis"]["range"][0],
            fig.layout["xaxis"]["range"][1] * 1.05,
        )
        m = len(variables_order)
        y_vals = [-5 - i * 10 for i in range(m)]
        fig.data[0]["y"] = y_vals

        ## triplot

        fig.add_shape(
            type="line",
            x0=0,
            x1=0,
            y0=-0.01,
            y1=1.01,
            yref="paper",
            xref="x2",
            line={"color": "#371ea3", "width": 1.5, "dash": "dot"},
        )

        min_x_imp, max_x_imp = np.inf, -np.inf
        for data in hierarchical_importance_dendrogram_plot["data"][::-1]:
            data["xaxis"] = "x2"
            data["hoverinfo"] = "text"
            data["line"] = {"color": "#46bac2", "width": 2}
            fig.add_trace(data)
            min_x_imp = np.min([min_x_imp, np.min(data["x"])])
            max_x_imp = np.max([max_x_imp, np.max(data["x"])])
        min_max_margin_imp = (max_x_imp - min_x_imp) * 0.15

        min_x_clust, max_x_clust = np.inf, -np.inf
        for data in hierarchical_clustering_dendrogram_plot["data"]:
            data["xaxis"] = "x3"
            data["hoverinfo"] = "text"
            data["line"] = {"color": "#46bac2", "width": 2}
            fig.add_trace(data)
            min_x_clust = np.min([min_x_clust, np.min(data["x"])])
            max_x_clust = np.max([max_x_clust, np.max(data["x"])])
        min_max_margin_clust = (max_x_clust - min_x_clust) * 0.15

        plot_height = 78 + 71 + m * bar_width + (m + 1) * bar_width / 4
        ticktext = plot.get_ticktext_for_plot(
            self.single_variable_importance, variables_order, abbrev_labels
        )

        fig.update_layout(
            xaxis={
                "autorange": False,
                "domain": [0, 0.33],
                "mirror": False,
                "showgrid": False,
                "showline": False,
                "zeroline": False,
                "ticks": "",
                "title_text": "Local variable importance",
            },
            xaxis2={
                "domain": [0.33, 0.66],
                "mirror": False,
                "showgrid": False,
                "showline": False,
                "zeroline": False,
                "showticklabels": True,
                "tickvals": [0],
                "ticktext": [""],
                "ticks": "",
                "title_text": "Hierarchical aspect importance",
                "autorange": False,
                "fixedrange": True,
                "range": [
                    min_x_imp - min_max_margin_imp,
                    max_x_imp + min_max_margin_imp,
                ],
            },
            xaxis3={
                "domain": [0.66, 0.99],
                "mirror": False,
                "showgrid": False,
                "showline": False,
                "zeroline": False,
                "showticklabels": True,
                "tickvals": [0],
                "ticktext": [""],
                "ticks": "",
                "title_text": "Hierarchical clustering",
                "autorange": False,
                "fixedrange": True,
                "range": [
                    min_x_clust - min_max_margin_clust,
                    max_x_clust + min_max_margin_clust,
                ],
            },
            yaxis={
                "mirror": False,
                "ticks": "",
                "fixedrange": True,
                "gridwidth": 1,
                "type": "linear",
                "tickmode": "array",
                "tickvals": y_vals,
                "ticktext": ticktext,
            },
            title_text=title,
            title_x=0.5,
            font={"color": "#371ea3"},
            template="none",
            margin={"t": 78, "b": 71, "r": 30},
            width=width,
            height=plot_height,
            showlegend=False,
            hovermode="closest"
        )

        fig, middle_point = plot._add_points_on_dendrogram_traces(fig)

        ##################################################################

        if widget:
            _global_checks.global_check_import('ipywidgets', 'Predict Triplot')
            from ipywidgets import HBox, Layout
            fig = go.FigureWidget(fig, layout={"autosize": True, "hoverdistance": 100})
            original_bar_colors = deepcopy(list(fig.data[0]["marker"]["color"]))
            original_text_colors = deepcopy(list(fig.data[0]["textfont"]["color"]))
            k = len(fig.data)
            updated_dendro_traces_in_full_figure = list(
                np.array(updated_dendro_traces) + (k - 1) / 2 + 1
            ) + list((k - 1) / 2 - np.array(updated_dendro_traces))

            def _update_childs(x, y, fig, k, selected, selected_y_cord):
                for i in range(1, k):
                    if middle_point[i] == (x, y):
                        fig.data[i]["line"]["color"] = "#46bac2"
                        fig.data[i]["line"]["width"] = 3
                        fig.data[k - i]["line"]["color"] = "#46bac2"
                        fig.data[k - i]["line"]["width"] = 3
                        selected.append(i)
                        selected.append(k - i)
                        if (fig.data[i]["y"][0] + 5) % 10 == 0:
                            selected_y_cord.append((fig.data[i]["y"][0] + 5) // -10)
                        if (fig.data[i]["y"][-1] - 5) % 10 == 0:
                            selected_y_cord.append((fig.data[i]["y"][-1] + 5) // -10)
                        _update_childs(
                            fig.data[i]["x"][0],
                            fig.data[i]["y"][0],
                            fig,
                            k,
                            selected,
                            selected_y_cord,
                        )
                        _update_childs(
                            fig.data[i]["x"][-1],
                            fig.data[i]["y"][-1],
                            fig,
                            k,
                            selected,
                            selected_y_cord,
                        )

            def _update_trace(trace, points, selector):
                if len(points.point_inds) == 1:
                    selected_ind = points.trace_index
                    with fig.batch_update():
                        if selected_ind not in updated_dendro_traces_in_full_figure:
                            for i in range(1, k):
                                fig.data[i]["line"]["color"] = "#46bac2"
                                fig.data[i]["line"]["width"] = 2
                                fig.data[i]["textfont"]["color"] = "#371ea3"
                                fig.data[i]["textfont"]["size"] = 12
                            fig.data[0]["marker"]["color"] = original_bar_colors
                            fig.data[0]["textfont"]["color"] = original_text_colors
                        else:
                            selected = [selected_ind, k - selected_ind]
                            selected_y_cord = []
                            if (fig.data[selected_ind]["y"][0] - 5) % 10 == 0:
                                selected_y_cord.append(
                                    (fig.data[selected_ind]["y"][0] + 5) // -10
                                )
                            if (fig.data[selected_ind]["y"][-1] - 5) % 10 == 0:
                                selected_y_cord.append(
                                    (fig.data[selected_ind]["y"][-1] + 5) // -10
                                )
                            fig.data[selected_ind]["line"]["color"] = "#46bac2"
                            fig.data[selected_ind]["line"]["width"] = 3
                            fig.data[selected_ind]["textfont"]["color"] = "#371ea3"
                            fig.data[selected_ind]["textfont"]["size"] = 14
                            fig.data[k - selected_ind]["line"]["color"] = "#46bac2"
                            fig.data[k - selected_ind]["line"]["width"] = 3
                            fig.data[k - selected_ind]["textfont"]["color"] = "#371ea3"
                            fig.data[k - selected_ind]["textfont"]["size"] = 14
                            _update_childs(
                                fig.data[selected_ind]["x"][0],
                                fig.data[selected_ind]["y"][0],
                                fig,
                                k,
                                selected,
                                selected_y_cord,
                            )
                            _update_childs(
                                fig.data[selected_ind]["x"][-1],
                                fig.data[selected_ind]["y"][-1],
                                fig,
                                k,
                                selected,
                                selected_y_cord,
                            )
                            for i in range(1, k):
                                if i not in [selected_ind, k - selected_ind]:
                                    fig.data[i]["textfont"]["color"] = "#ceced9"
                                    fig.data[i]["textfont"]["size"] = 12
                                    if i not in selected:
                                        fig.data[i]["line"]["color"] = "#ceced9"
                                        fig.data[i]["line"]["width"] = 1

                            bars_colors_list = deepcopy(original_bar_colors)
                            text_colors_list = deepcopy(original_text_colors)
                            for i in range(m):
                                if i not in selected_y_cord:
                                    bars_colors_list[i] = "#ceced9"
                                    text_colors_list[i] = "#ceced9"
                            fig.data[0]["marker"]["color"] = bars_colors_list
                            fig.data[0]["textfont"]["color"] = text_colors_list

            for i in range(1, k):
                fig.data[i].on_click(_update_trace)
            return HBox([fig], layout=Layout(overflow='scroll', width=f'{fig.layout.width}px'))
        if show:
            fig.show(config=_theme.get_default_config())
        else:
            return fig
