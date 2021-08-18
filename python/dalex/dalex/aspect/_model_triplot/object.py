from copy import deepcopy

import pandas as pd
from dalex.aspect._model_aspect_importance.object import ModelAspectImportance

import numpy as np
import plotly.graph_objs as go

from dalex import _theme
from dalex._explanation import Explanation
from dalex.model_explanations._variable_importance import VariableImportance
from . import checks, utils, plot


class ModelTriplot(Explanation):
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
        self.hierarchical_importance_data = pd.DataFrame()
        self.single_variable_importance_data = None
        self.hierarchical_clustering_dendrogram = None

    def _repr_html_(self):
        return self.hierarchical_importance_data._repr_html_()

    def fit(self, aspect):
        self.hierarchical_clustering_dendrogram = (
            aspect._hierarchical_clustering_dendrogram
        )

        (
            self.hierarchical_importance_data,
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

        self.hierarchical_importance_data.insert(3, "min_depend", None)
        self.hierarchical_importance_data.insert(4, "vars_min_depend", None)
        for index, row in self.hierarchical_importance_data.iterrows():
            _matching_row = aspect._dendrogram_aspects_ordered.loc[
                pd.Series(map(set, aspect._dendrogram_aspects_ordered.variables_names))
                == set(row.variables_names)
            ]
            min_dep = _matching_row.min_depend.values[0]
            vars_min_depend = _matching_row.vars_min_depend.values[0]
            self.hierarchical_importance_data.at[index, "min_depend"] = min_dep
            self.hierarchical_importance_data.at[
                index, "vars_min_depend"
            ] = vars_min_depend
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
        self.single_variable_importance_data = self._variable_importance_object.result

        single_vi = deepcopy(self.single_variable_importance_data)
        single_vi["h"] = 1
        aspect._full_hierarchical_aspect_importance = pd.concat(
            [aspect._full_hierarchical_aspect_importance, single_vi]
        )

    def plot(
        self,
        rounding_function=np.around,
        digits=3,
        bar_width=25,
        width=1500,
        title="Model Triplot",
        show=True,
        widget=False,
    ):
        ## right plot
        hierarchical_clustering_dendrogram_plot_without_annotations = (
            self.hierarchical_clustering_dendrogram
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
            self.hierarchical_importance_data,
            rounding_function,
            digits,
        )

        hierarchical_clustering_dendrogram_plot = plot.add_text_to_dendrogram(
            hierarchical_clustering_dendrogram_plot_without_annotations,
            updated_dendro_traces,
            type="clustering",
        )

        hierarchical_importance_dendrogram_plot = plot.add_text_to_dendrogram(
            hierarchical_importance_dendrogram_plot_without_annotations,
            updated_dendro_traces,
            type="importance",
        )

        ## left plot
        _result = deepcopy(self.single_variable_importance_data)
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

        ## Do not touch! It works, but we don't know why.
        fig.layout["shapes"][0]["y0"] = -0.01
        fig.layout["shapes"][0]["y1"] = 0.1
        ##
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

        min_x_clust, max_x_clust = np.Inf, -np.Inf
        for data in hierarchical_clustering_dendrogram_plot["data"]:
            data["xaxis"] = "x3"
            data["hoverinfo"] = "text"
            data["line"] = {"color": "#46bac2", "width": 2}
            fig.add_trace(data)
            min_x_clust = np.min(
                [
                    min_x_clust,
                    np.min(data["x"]),
                ]
            )
            max_x_clust = np.max(
                [
                    max_x_clust,
                    np.max(data["x"]),
                ]
            )

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
            }
        )

        min_max_margin_imp = (max_x_imp - min_x_imp) * 0.15
        fig.update_layout(
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
            }
        )
        min_max_margin_clust = (max_x_clust - min_x_clust) * 0.15
        fig.update_layout(
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
            }
        )

        fig.update_layout(
            yaxis={
                "mirror": False,
                "ticks": "",
                "fixedrange": True,
                "gridwidth": 1,
                "type": "linear",
                "tickmode": "array",
                "tickvals": y_vals,
                "ticktext": variables_order,
            }
        )

        plot_height = 78 + 71 + m * bar_width + (m + 1) * bar_width / 4
        fig.update_layout(
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
            fig.update_layout(width=None)
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
            return fig
        if show:
            fig.show(config=_theme.get_default_config())
        else:
            return fig
