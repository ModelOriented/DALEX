import numpy as np
import pandas as pd
from copy import deepcopy
import plotly.graph_objs as go

from dalex import _theme, _global_checks
from dalex._explanation import Explanation
from dalex.aspect._model_aspect_importance.object import ModelAspectImportance

from . import checks, plot, utils


class ModelTriplot(Explanation):
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

    Attributes
    -----------
    result : pd.DataFrame
        Main result attribute of an explanation.
    single_variable_importance : pd.DataFrame
        Additional result attribute of an explanation (it contains information 
        about the importance of individual variables).
    loss_function : function
        Loss function used to assess the variable importance.
    type : {'variable_importance', 'ratio', 'difference'}
        Type of transformation that will be applied to dropout loss.
    N : int
        Number of observations that will be sampled from the `explainer.data` attribute before
        the calculation of aspect importance. `None` means all `data` (default is `1000`).
    B : int
        Number of permutation rounds to perform on each variable (default is `10`).
    processes : int
        Number of parallel processes to use in calculations. Iterated over `B`
        (default is `1`, which means no parallel computation).
    random_state : int or None
        Set seed for random number generator.
    """
    def __init__(
        self,
        loss_function="rmse",
        type="variable_importance",
        N=1000,
        B=10,
        processes=1,
        random_state=None,
    ):
        _B = checks.check_B(B)
        _type = checks.check_type(type)
        _random_state = checks.check_random_state(random_state)
        _processes = checks.check_processes(processes)

        self.loss_function = loss_function
        self.type = _type
        self.N = N
        self.B = _B
        self.random_state = _random_state
        self.processes = _processes
        self.result = pd.DataFrame()
        self.single_variable_importance = None
        self._hierarchical_clustering_dendrogram = None

    def _repr_html_(self):
        return self.result._repr_html_()

    def fit(self, aspect):
        """Calculate the result of explanation
        Fit method makes calculations in place and changes the attributes.

        Parameters
        ----------
        aspect : Aspect object
            Explainer wrapper created using the Aspect class.
        
        Returns
        -----------
        None
        """

        self._hierarchical_clustering_dendrogram = (
            aspect._hierarchical_clustering_dendrogram
        )

        (
            self.result,
            aspect._full_hierarchical_aspect_importance,
        ) = utils.calculate_model_hierarchical_importance(
            aspect,
            self.loss_function,
            self.type,
            self.N,
            self.B,
            self.processes,
            self.random_state,
        )

        self.result.insert(3, "min_depend", None)
        self.result.insert(4, "vars_min_depend", None)
        for index, row in self.result.iterrows():
            _matching_row = aspect._dendrogram_aspects_ordered.loc[
                pd.Series(map(set, aspect._dendrogram_aspects_ordered.variables_names))
                == set(row.variables_names)
            ]
            min_dep = _matching_row.min_depend.values[0]
            vars_min_depend = _matching_row.vars_min_depend.values[0]
            self.result.at[index, "min_depend"] = min_dep
            self.result.at[index, "vars_min_depend"] = vars_min_depend
        ## left plot data
        variable_groups = aspect.get_aspects(h=2)
        self._variable_importance_object = ModelAspectImportance(
            variable_groups=variable_groups,
            loss_function=self.loss_function,
            type=self.type,
            N=self.N,
            B=self.B,
            processes=self.processes,
            random_state=self.random_state,
            _is_aspect_model_parts=False,
        )
        self._variable_importance_object.fit(aspect.explainer)
        self.single_variable_importance = self._variable_importance_object.result

        single_vi = deepcopy(self.single_variable_importance)
        single_vi["h"] = 1
        aspect._full_hierarchical_aspect_importance = pd.concat(
            [aspect._full_hierarchical_aspect_importance, single_vi]
        )

    def plot(
        self,
        digits=3,
        rounding_function=np.around,
        bar_width=25,
        width=1500,
        title="Model Triplot",
        widget=False,
        show=True
    ):
        """Plot the Model Triplot explanation (triplot visualization).

        Parameters
        ----------
        digits : int, optional
            Number of decimal places (`np.around`) to round contributions.
            See `rounding_function` parameter (default is `3`).
        rounding_function : function, optional
            A function that will be used for rounding numbers (default is `np.around`).
        bar_width : float, optional
            Width of bars in px (default is `16`).
        width : float, optional
            Width of triplot in px (default is `1500`).
        title : str, optional
            Title of the plot (default is `"Model Triplot"`).
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
        _global_checks.global_check_import('kaleido', 'Model Triplot')
        ## right plot
        hierarchical_clustering_dendrogram_plot_without_annotations = (
            self._hierarchical_clustering_dendrogram
        )
        variables_order = list(
            hierarchical_clustering_dendrogram_plot_without_annotations.layout.yaxis.ticktext
        )
        m = len(variables_order)

        ## middle plot
        (
            hierarchical_importance_dendrogram_plot_without_annotations,
            updated_dendro_traces,
        ) = plot.plot_model_hierarchical_importance(
            hierarchical_clustering_dendrogram_plot_without_annotations,
            self.result,
            rounding_function,
            digits,
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
        _result = deepcopy(self.single_variable_importance)
        sorter = dict(zip(variables_order, range(m)))
        _result["order"] = _result["aspect_name"].map(sorter)
        _result = _result.sort_values(["order"], ascending=True).reset_index(drop=True)
        _result = _result.drop("order", axis=1)
        self._variable_importance_object.result = _result

        fig = self._variable_importance_object.plot(
            max_aspects=_result.shape[0],
            digits=digits,
            rounding_function=rounding_function,
            bar_width=bar_width,
            show=False,
            title=None,
            show_variables_names=False
        )
        fig.data[0]["textfont_color"] = ["#371ea3"] * m

        fig.layout["shapes"][0]["y0"] = -0.01
        fig.layout["shapes"][0]["y1"] = 1.01
        fig.layout["shapes"][0]["yref"] = 'paper'
        
        fig.layout["xaxis"]["range"] = (
            fig.layout["xaxis"]["range"][0],
            fig.layout["xaxis"]["range"][1] * 1.05,
        )
        y_vals = [-5 - i * 10 for i in range(m)]
        fig.data[0]["y"] = y_vals

        ## triplot
        min_x_imp, max_x_imp = np.Inf, -np.Inf
        for data in hierarchical_importance_dendrogram_plot["data"][::-1]:
            data["xaxis"] = "x2"
            data["hoverinfo"] = "text"
            data["line"] = {"color": "#46bac2", "width": 2}
            fig.add_trace(data)
            min_x_imp = np.min([min_x_imp, np.min(data["x"])])
            max_x_imp = np.max([max_x_imp, np.max(data["x"])])
        min_max_margin_imp = (max_x_imp - min_x_imp) * 0.15

        min_x_clust, max_x_clust = np.Inf, -np.Inf
        for data in hierarchical_clustering_dendrogram_plot["data"]:
            data["xaxis"] = "x3"
            data["hoverinfo"] = "text"
            data["line"] = {"color": "#46bac2", "width": 2}
            fig.add_trace(data)
            min_x_clust = np.min([min_x_clust, np.min(data["x"])])
            max_x_clust = np.max([max_x_clust, np.max(data["x"])])
        min_max_margin_clust = (max_x_clust - min_x_clust) * 0.15

        plot_height = 78 + 71 + m * bar_width + (m + 1) * bar_width / 4

        fig.update_layout(
            xaxis={
                "autorange": False,
                "domain": [0, 0.33],
                "mirror": False,
                "showgrid": False,
                "showline": False,
                "zeroline": False,
                "showticklabels": True,
                "ticks": "",
                "title_text": "Variable importance",
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
                "fixedrange": True,
                "autorange": False,
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
                "fixedrange": True,
                "autorange": False,
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
                "ticktext": variables_order,
            },
            title_text=title,
            title_x=0.5,
            font={"color": "#371ea3"},
            template="none",
            margin={"t": 78, "b": 71, "r": 30},
            width=width,
            height=plot_height,
            showlegend=False,
            hovermode="closest",

        )

        fig, middle_point = plot._add_points_on_dendrogram_traces(fig)

        ##################################################################

        if widget:
            from ipywidgets import HBox, Layout
            fig = go.FigureWidget(fig, layout={"autosize": True, "hoverdistance": 100})
            original_bar_colors = deepcopy([fig.data[0]["marker"]["color"]] * m)
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
                        if max(fig.data[selected_ind]["x"]) in (max_x_clust, max_x_imp):
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
