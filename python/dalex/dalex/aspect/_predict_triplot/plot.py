import numpy as np
import plotly.graph_objects as go
from copy import deepcopy 

from dalex import _global_utils, _theme

from . import utils



def plot_predict_hierarchical_importance(
    dendrogram_hierarchical_correlation,
    hierarchical_importance_data,
    rounding_function,
    digits,
    absolute_value,
    type
):
    dendrogram_hierarchical_importance = go.Figure(dendrogram_hierarchical_correlation)
    importance_dict = {}
    min_imp = min(hierarchical_importance_data["importance"])
    max_imp = max(hierarchical_importance_data["importance"])


    tick_dict = dict(zip(dendrogram_hierarchical_importance.layout.yaxis.tickvals, 
                    [[var] for var in dendrogram_hierarchical_importance.layout.yaxis.ticktext]))
    aspects_dendro_order = []
    for scatter in dendrogram_hierarchical_importance.data:
        tick_dict[np.mean(scatter.y[1:3])] = tick_dict[scatter.y[1]] + tick_dict[scatter.y[2]]
        aspects_dendro_order.append(tick_dict[scatter.y[1]] + tick_dict[scatter.y[2]])

    updated = []
    for i, row in hierarchical_importance_data[:-1].iterrows():
        dendro_mask = np.array([set(row.variable_names) == set(el) for el in aspects_dendro_order])
        if dendro_mask.any():
            k = np.flatnonzero(dendro_mask == True)[0]
            updated.append(k)
            imp_scatter = row["importance"]
            if absolute_value:
                imp_scatter = abs(imp_scatter)
            scatter_importance = dendrogram_hierarchical_importance.data[k]
            scatter_clustering = dendrogram_hierarchical_correlation.data[k]
            importance_dict[np.mean(scatter_importance.y[1:3])] = imp_scatter
            scatter_importance.x[1:3] = imp_scatter

            scatter_importance.hoverinfo = "text"
            scatter_importance.hoverlabel = {"bgcolor": "rgba(0,0,0,0.8)"}
            scatter_importance.hovertext = tooltip_text_aspect(row, rounding_function, digits)

            scatter_clustering.hoverinfo = "text"
            scatter_clustering.hoverlabel = {"bgcolor": "rgba(0,0,0,0.8)"}
            scatter_clustering.hovertext = tooltip_text_aspect(row, rounding_function, digits)

    for i, scatter in enumerate(dendrogram_hierarchical_importance.data):
        if i not in updated:
            if absolute_value == True:
                imp_scatter = abs(hierarchical_importance_data.iloc[-1]["importance"])
            else:
                imp_scatter = hierarchical_importance_data.iloc[-1]["importance"]
            change_last = True if type == 'default' else False
            scatter.hoverinfo = "text"
            scatter.hoverlabel = {"bgcolor": "rgba(0,0,0,0.8)"}
            scatter.hovertext = tooltip_text_aspect(hierarchical_importance_data.iloc[-1], rounding_function, digits, change_last)
            scatter_clustering = dendrogram_hierarchical_correlation.data[i]
            scatter_clustering.hoverinfo = "text"
            scatter_clustering.hoverlabel = {"bgcolor": "rgba(0,0,0,0.8)"}
            scatter_clustering.hovertext = tooltip_text_aspect(hierarchical_importance_data.iloc[-1], rounding_function, digits, change_last)
            scatter.x[1:3] = imp_scatter
            importance_dict[np.mean(scatter.y[1:3])] = imp_scatter
        if scatter.y[0] in importance_dict.keys():
            scatter.x[0] = importance_dict[scatter.y[0]]
        if scatter.y[3] in importance_dict.keys():
            scatter.x[3] = importance_dict[scatter.y[3]]

    min_range = (
        min_imp * 1.15
        if min_imp < 0
        else min_imp * 0.85
    )
    max_range = (
        max_imp * 1.15
        if max_imp * 1.15 > 0
        else max_imp * 0.85
    )
    dendrogram_hierarchical_importance.update_xaxes(
        range=[min_range, max_range],
        visible=False,
        showticklabels=False,
    )
    return dendrogram_hierarchical_importance, updated


def get_ticktext_for_plot(
    single_aspect_importance_df, variables_order, abbrev_labels=0
):
    _df = deepcopy(single_aspect_importance_df)
    ticktext = [
        f"{variable} = {utils.nice_format(_df.loc[_df['variable_names']==variable, 'variable_values'].values[0])}"
        if abbrev_labels <= 0
        else utils.text_abbreviate(
            f"{variable} = {utils.nice_format(_df.loc[_df['variable_names']==variable, 'variable_values'].values[0])}",
            abbrev_labels,
        )
        for variable in variables_order
    ]
    return ticktext


def tooltip_text_aspect(row, rounding_function, digits, last=False):
    var_val_string = ""
    for i in range(len(row.variable_names)):
        var_val_string += (
            "<br>" + row.variable_names[i] + " = " + str(row.variable_values[i])
        )
    keyword = "Change in average response: " if last else "Importance: " 

    return (
        f"Min abs depend: {rounding_function(row.min_depend, digits)}<br>"
        + "(between variables: " + ", ".join(row.vars_min_depend) + ")<br>"
        + keyword + f"{rounding_function(row.importance, digits)}<br>Variables: "
        + var_val_string
    )


def tooltip_text_single(row, rounding_function, digits):
    return (
        f"Importance: {rounding_function(row.importance, digits)}<br>"
        + "Variable:<br>"
        + f"{row.variable_names} = " + str(row.variable_values)
    )


def add_text_to_dendrogram(fig, updated, rounding_function, digits, type="clustering"):
    res_fig = go.Figure(fig)
    corner_scatters = [] 
    for i, scatter in enumerate(res_fig.data):
        if i in updated:
            x_cord = scatter.x[1]
            y_cord = np.mean(scatter.y[1:3])
            if type == "clustering":
                lab_text = str(rounding_function(1-x_cord, digits))
            else:
                lab_text = str(rounding_function(x_cord, digits))
            scatter.x = np.insert(scatter.x, 2, x_cord)
            scatter.y = np.insert(scatter.y, 2, y_cord)
            scatter.mode = "text+lines"
            scatter["text"] = [None, None, lab_text, None, None]
            scatter["textposition"] = "middle left"
        else:
            corner_scatters.append(scatter)
    if corner_scatters:
        scatter = corner_scatters[len(corner_scatters)//2]
        x_cord = scatter.x[1]
        y_cord = np.mean(scatter.y[1:3])
        if type == "clustering":
            lab_text = str(rounding_function(1-x_cord, digits))
        else:
            lab_text = str(rounding_function(x_cord, digits))
        scatter.x = np.insert(scatter.x, 2, x_cord)
        scatter.y = np.insert(scatter.y, 2, y_cord)
        scatter.mode = "text+lines"
        scatter["text"] = [None, None, lab_text, None, None]
        scatter["textposition"] = "middle left"

    return res_fig


def plot_single_aspects_importance(
    result_dataframe, order, rounding_function, digits, vcolors
):
    fig = go.Figure()
    _result = deepcopy(result_dataframe)
    sorter = dict(zip(order, range(len(order))))
    _result["order"] = _result["variable_names"].map(sorter)
    _result = _result.sort_values(["order"], ascending=True).reset_index(drop=True)
    _result = _result.drop("order", axis=1)
    _result.loc[:, "importance"] = rounding_function(
                _result.loc[:, "importance"], digits
            )

    _result["color"] = [0 if imp > 0 else 1 for imp in _result["importance"]]
    _result["tooltip_text"] = _result.apply(tooltip_text_single, args=(rounding_function, digits), axis=1)
    _result["label_text"] = _global_utils.convert_float_to_str(_result.importance, "+")

    if vcolors is None:
        vcolors = _theme.get_aspect_importance_colors()

    fig.add_shape(
        type="line",
        x0=0,
        x1=0,
        y0=-0.01,
        y1=1.01,
        yref="paper",
        xref="x",
        line={"color": "#371ea3", "width": 1.5, "dash": "dot"},
    )

    fig.add_bar(
        orientation="h",
        y=_result["variable_names"].tolist(),
        x=_result["importance"].tolist(),
        textposition="outside",
        text=_result["label_text"].tolist(),
        textfont_color=["#371ea3"] * len(order),
        marker_color=[vcolors[int(c)] for c in _result["color"].tolist()],
        hovertext=_result["tooltip_text"].tolist(),
        hoverinfo="text",
        hoverlabel={"bgcolor": "rgba(0,0,0,0.8)"},
        showlegend=False,
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
        }
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
        }
    )

    temp_min_max = [np.Inf, -np.Inf]
    min_max_margin = _result.importance.values.ptp() * 0.15
    temp_min_max[0] = np.min(
        [temp_min_max[0], _result.importance.values.min() - min_max_margin]
    )
    temp_min_max[1] = np.max(
        [temp_min_max[1], _result.importance.values.max() + min_max_margin]
    )

    fig.update_xaxes({"range": temp_min_max})
    fig.update_layout(
        title_x=0.15,
        font={"color": "#371ea3"},
        template="none",
        margin={"t": 78, "b": 71, "r": 30},
    )

    return fig


def _add_between_points(
        arr, arr_2, val_arr_2, range_global, i_begin, i_end, arr_text=None
    ):
        curr_val = arr[i_begin]
        last_val = arr[i_end]
        range_curr = abs(curr_val - last_val)
        num = int((range_curr * 40) / range_global)
        num = max(num, 5)
        new_points = np.linspace(
            curr_val,
            last_val,
            num=num,
        )
        i_curr = i_begin + 1
        arr = np.insert(arr, i_curr, new_points[1:-1])
        arr_2 = np.insert(arr_2, i_curr, [val_arr_2] * (num - 2))
        if arr_text is not None:
            arr_text = np.insert(arr_text, i_curr, [None] * (num - 2))
        return arr, arr_2, num - 2, arr_text

def _add_points_on_dendrogram_traces(fig):
    k = len(fig.data)
    range_x2 = fig.full_figure_for_development(warn=False).layout.xaxis2.range
    range_x2_len = range_x2[1] - range_x2[0]
    range_x3 = fig.full_figure_for_development(warn=False).layout.xaxis3.range
    range_x3_len = range_x3[1] - range_x3[0]
    range_y = fig.full_figure_for_development(warn=False).layout.yaxis.range
    range_y_len = range_y[1] - range_y[0]
    middle_point = [None] * k
    for i in range(1, k):
        if i < ((k - 1) / 2 + 1):
            range_x_len = range_x2_len
        else:
            range_x_len = range_x3_len
        arr_text = None
        num_points = len(fig.data[i]["x"])
        if num_points == 5:
            arr_text = fig.data[i]["text"]
        middle_point[i] = fig.data[i]["x"][2], fig.data[i]["y"][2]
        inserted_points = 0
        fig.data[i]["x"], fig.data[i]["y"], j, arr_text = _add_between_points(
            fig.data[i]["x"],
            fig.data[i]["y"],
            fig.data[i]["y"][inserted_points],
            range_x_len,
            inserted_points,
            inserted_points + 1,
            arr_text,
        )
        inserted_points += j + 1
        fig.data[i]["y"], fig.data[i]["x"], j, arr_text = _add_between_points(
            fig.data[i]["y"],
            fig.data[i]["x"],
            fig.data[i]["x"][inserted_points],
            range_y_len,
            inserted_points,
            inserted_points + 1,
            arr_text,
        )
        inserted_points += j + 1
        if num_points == 5:
            (
                fig.data[i]["y"],
                fig.data[i]["x"],
                j,
                arr_text,
            ) = _add_between_points(
                fig.data[i]["y"],
                fig.data[i]["x"],
                fig.data[i]["x"][inserted_points],
                range_y_len,
                inserted_points,
                inserted_points + 1,
                arr_text,
            )
            inserted_points += j + 1
        fig.data[i]["x"], fig.data[i]["y"], j, arr_text = _add_between_points(
            fig.data[i]["x"],
            fig.data[i]["y"],
            fig.data[i]["y"][inserted_points],
            range_x_len,
            inserted_points,
            inserted_points + 1,
            arr_text,
        )
        inserted_points += j + 1
        
        if arr_text is not None:
            fig.data[i]["text"] = arr_text
    return fig, middle_point

