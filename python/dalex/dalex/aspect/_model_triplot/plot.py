from .._plotting import as_plot_data, ordered
import numpy as np
from ..plot import copy_dendrogram
import plotly.graph_objects as go
from copy import deepcopy 

from dalex import _theme, _global_utils


def plot_model_hierarchical_importance(
    dendrogram_hierarchical_correlation,
    hierarchical_importance_data,
    rounding_function,
    digits,
    change
):
    dendrogram_hierarchical_importance = copy_dendrogram(dendrogram_hierarchical_correlation)
    hierarchical_importance_data = as_plot_data(hierarchical_importance_data)
    hierarchical_rows = hierarchical_importance_data.rows()
    importance_dict = {}
    max_imp = max(hierarchical_importance_data.frame["dropout_loss_change"].to_numpy()) if change else max(hierarchical_importance_data.frame["dropout_loss"].to_numpy())

    tick_dict = dict(
        zip(
            dendrogram_hierarchical_importance.layout.yaxis.tickvals,
            [[var] for var in dendrogram_hierarchical_importance.layout.yaxis.ticktext],
        )
    )
    aspects_dendro_order = []
    aspects_vars_before = []
    for i, scatter in enumerate(dendrogram_hierarchical_importance.data):
        tick_dict[np.mean(scatter.y[1:3])] = (
            tick_dict[scatter.y[1]] + tick_dict[scatter.y[2]]
        )
        aspects_dendro_order.append(tick_dict[scatter.y[1]] + tick_dict[scatter.y[2]])
        aspects_vars_before.append((tick_dict[scatter.y[1]], tick_dict[scatter.y[2]]))

    updated = []
    for i, row in enumerate(hierarchical_rows):
        dendro_mask = np.array(
            [set(row['variable_names']) == set(el) for el in aspects_dendro_order]
        )
        if dendro_mask.any():
            k = np.flatnonzero(dendro_mask == True)[0]
            updated.append(k)
            imp_scatter = row["dropout_loss_change"] if change else row["dropout_loss"]
            scatter_importance = dendrogram_hierarchical_importance.data[k]
            scatter_clustering = dendrogram_hierarchical_correlation.data[k]
            importance_dict[tuple(aspects_dendro_order[k])] = imp_scatter
            scatter_importance.x[1:3] = imp_scatter

            scatter_importance.hoverinfo = "text"
            scatter_importance.hoverlabel = {"bgcolor": "rgba(0,0,0,0.8)"}
            scatter_importance.hovertext = tooltip_text(
                row, rounding_function, digits
            )

            scatter_clustering.hoverinfo = "text"
            scatter_clustering.hoverlabel = {"bgcolor": "rgba(0,0,0,0.8)"}
            scatter_clustering.hovertext = tooltip_text(
                row, rounding_function, digits
            )

    for i, scatter in enumerate(dendrogram_hierarchical_importance.data):
        vars_before = aspects_vars_before[i]
        if i not in updated:
            scatter.x[1:3] = max_imp
            importance_dict[tuple(aspects_dendro_order[i])] = max_imp
        if tuple(vars_before[0]) in importance_dict.keys():
            scatter.x[0] = importance_dict[tuple(vars_before[0])]
        if tuple(vars_before[1]) in importance_dict.keys():
            scatter.x[3] = importance_dict[tuple(vars_before[1])]

    dendrogram_hierarchical_importance.update_xaxes(
        range=[0, max(hierarchical_importance_data.frame["dropout_loss_change"].to_numpy()) * 1.05] if change
                else [0, max(hierarchical_importance_data.frame["dropout_loss"].to_numpy()) * 1.05],
        visible=False,
        showticklabels=False,
    )
    return dendrogram_hierarchical_importance, updated


def plot_single_aspects_importance(
    result_dataframe, order, rounding_function, digits, vcolors
):
    fig = go.Figure()
    original = as_plot_data(result_dataframe)
    baseline = original.frame['dropout_loss'][0] - original.frame['dropout_loss_change'][0]
    _data = ordered(original, order).rounded('dropout_loss_change', rounding_function, digits)
    _rows = _data.rows()
    _result = _data.frame
    tooltip = [tooltip_text_single(row, rounding_function, digits) for row in _rows]
    labels = _global_utils.convert_float_to_str(_result['dropout_loss_change'].to_numpy(), '+')

    if vcolors is None:
        vcolors = _theme.get_default_colors(1, 'bar')[0]

    fig.add_shape(
        type="line",
        x0=baseline,
        x1=baseline,
        y0=-0.01,
        y1=1.01,
        yref="paper",
        xref="x",
        line={"color": "#371ea3", "width": 1.5, "dash": "dot"},
    )

    fig.add_bar(
        orientation="h",
        y=[row['variable_names'] for row in _rows],
        x=_result["dropout_loss_change"].to_list(),
        base=baseline,
        textposition="outside",
        text=labels,
        textfont_color=["#371ea3"] * len(order),
        marker_color=vcolors,
        hovertext=tooltip,
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

    temp_min_max = [np.inf, -np.inf]
    min_max_margin = np.ptp(_result['dropout_loss'].to_numpy()) * 0.15
    temp_min_max[0] = np.min(
        [temp_min_max[0], baseline - min_max_margin]
    )
    temp_min_max[1] = np.max(
        [temp_min_max[1], _result['dropout_loss'].to_numpy().max() + min_max_margin]
    )

    fig.update_xaxes({"range": temp_min_max})
    fig.update_layout(
        title_x=0.15,
        font={"color": "#371ea3"},
        template="none",
        margin={"t": 78, "b": 71, "r": 30},
    )

    return fig


def add_text_to_model_importance_dendrogram(fig, hierarchical_importance_data):
    hierarchical_importance_data = as_plot_data(hierarchical_importance_data)
    res_fig = copy_dendrogram(fig)
    for scatter in res_fig.data:
        k = np.where(hierarchical_importance_data.frame["dropout_loss"].to_numpy() == scatter.x[1])[0][0]
        x_cord = scatter.x[1]
        y_cord = np.mean(scatter.y[1:3])
        scatter.x = np.insert(scatter.x, 2, x_cord)
        scatter.y = np.insert(scatter.y, 2, y_cord)
        label = "{0:.2f}".format(hierarchical_importance_data.frame["dropout_loss_change"].to_numpy()[k])
        scatter.mode = "lines+text"
        scatter["text"] = [None, None, label, None, None]
        scatter["textposition"] = "middle left"

    return res_fig


def tooltip_text(row, rounding_function, digits):

    if row['dropout_loss_change'] > 0:
        dropout_loss_change_string = "+" + str(
            rounding_function(row['dropout_loss_change'], digits)
        )
    else:
        dropout_loss_change_string = str(
            rounding_function(row['dropout_loss_change'], digits)
        )
    return (
        f"Min abs depend: {rounding_function(row['min_depend'], digits)}<br>"
        + "(between variables: "
        + ", ".join(row['vars_min_depend'])
        + ")<br>"
        + f"Drop-out loss: {rounding_function(row['dropout_loss'], digits)}<br>"
        + "Drop-out loss change: "
        + dropout_loss_change_string
        + "<br>"
        + "Variables:<br>"
        + "<br>".join(row['variable_names'])
    )


def tooltip_text_single(row, rounding_function, digits):
    if row['dropout_loss_change'] > 0:
        key_word = "+"
    else:
        key_word = "-"
    return (
        "Drop-out loss: "
        + str(rounding_function(row['dropout_loss'], digits))
        + "<br>"
        + "Drop-out loss change: "
        + key_word
        + str(rounding_function(np.abs(row['dropout_loss_change']), digits))
        + "<br>"
        + "Variable: "
        + str(row['variable_names'])
    )


def add_text_to_dendrogram(fig, updated_dendro_traces, rounding_function, digits, type="clustering"):
    res_fig = copy_dendrogram(fig)
    for i, scatter in enumerate(res_fig.data):
        if i in updated_dendro_traces:
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
    # Only hover-point density depends on these spans. All added points remain
    # on existing straight segments, so resolving browser defaults is unnecessary.
    range_x2_len = _global_utils.finite_range_length(fig.layout.xaxis2.range)
    range_x3_len = _global_utils.finite_range_length(fig.layout.xaxis3.range)
    range_y_len = _global_utils.finite_range_length([
        value for trace in fig.data if trace.y is not None for value in trace.y
    ])
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
