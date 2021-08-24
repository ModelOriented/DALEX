from dalex.aspect._model_aspect_importance.object import ModelAspectImportance
from dalex.model_explanations._variable_importance.object import VariableImportance
import numpy as np
import plotly.graph_objects as go


def plot_model_hierarchical_importance(
    dendrogram_hierarchical_correlation,
    hierarchical_importance_data,
    rounding_function,
    digits,
):
    dendrogram_hierarchical_importance = go.Figure(dendrogram_hierarchical_correlation)
    importance_dict = {}
    max_imp = max(hierarchical_importance_data["dropout_loss"])

    tick_dict = dict(
        zip(
            dendrogram_hierarchical_importance.layout.yaxis.tickvals,
            [[var] for var in dendrogram_hierarchical_importance.layout.yaxis.ticktext],
        )
    )
    aspects_dendro_order = []
    for scatter in dendrogram_hierarchical_importance.data:
        tick_dict[np.mean(scatter.y[1:3])] = (
            tick_dict[scatter.y[1]] + tick_dict[scatter.y[2]]
        )
        aspects_dendro_order.append(tick_dict[scatter.y[1]] + tick_dict[scatter.y[2]])

    updated = []
    for i, row in hierarchical_importance_data.iterrows():
        dendro_mask = np.array(
            [set(row.variables_names) == set(el) for el in aspects_dendro_order]
        )
        if dendro_mask.any():
            k = np.flatnonzero(dendro_mask == True)[0]
            updated.append(k)
            imp_scatter = row["dropout_loss"]
            scatter_importance = dendrogram_hierarchical_importance.data[k]
            scatter_clustering = dendrogram_hierarchical_correlation.data[k]
            importance_dict[np.mean(scatter_importance.y[1:3])] = imp_scatter
            scatter_importance.x[1:3] = imp_scatter

            scatter_importance.hoverinfo = "text"
            scatter_importance.hoverlabel = {"bgcolor": "rgba(0,0,0,0.8)"}
            scatter_importance.hovertext = get_tooltip_text(
                row, rounding_function, digits
            )

            scatter_clustering.hoverinfo = "text"
            scatter_clustering.hoverlabel = {"bgcolor": "rgba(0,0,0,0.8)"}
            scatter_clustering.hovertext = get_tooltip_text(
                row, rounding_function, digits
            )

    for i, scatter in enumerate(dendrogram_hierarchical_importance.data):
        if i not in updated:
            scatter.x[1:3] = max_imp
            importance_dict[np.mean(scatter.y[1:3])] = max_imp

    for scatter in dendrogram_hierarchical_importance.data:
        if scatter.y[0] in importance_dict.keys():
            scatter.x[0] = importance_dict[scatter.y[0]]
        if scatter.y[3] in importance_dict.keys():
            scatter.x[3] = importance_dict[scatter.y[3]]

    dendrogram_hierarchical_importance.update_xaxes(
        range=[0, max(hierarchical_importance_data["dropout_loss"]) * 1.05],
        visible=False,
        showticklabels=False,
    )
    return dendrogram_hierarchical_importance, updated


def add_text_to_model_importance_dendrogram(fig, hierarchical_importance_data):
    res_fig = go.Figure(fig)
    for scatter in res_fig.data:
        k = np.where(hierarchical_importance_data["dropout_loss"] == scatter.x[1])[0][0]
        x_cord = scatter.x[1]
        y_cord = np.mean(scatter.y[1:3])
        scatter.x = np.insert(scatter.x, 2, x_cord)
        scatter.y = np.insert(scatter.y, 2, y_cord)
        label = "{0:.2f}".format(hierarchical_importance_data["dropout_loss_change"][k])
        scatter.mode = "lines+text"
        scatter["text"] = [None, None, label, None, None]
        scatter["textposition"] = "middle left"

    return res_fig


def get_tooltip_text(row, rounding_function, digits):

    if row.dropout_loss_change > 0:
        dropout_loss_change_string = "+" + str(
            rounding_function(row.dropout_loss_change, digits)
        )
    else:
        dropout_loss_change_string = str(
            rounding_function(row.dropout_loss_change, digits)
        )
    return (
        f"Min abs depend: {rounding_function(row.min_depend, digits)}<br>"
        + "(between variables: "
        + ", ".join(row.vars_min_depend)
        + ")<br>"
        + f"Drop-out loss: {rounding_function(row.dropout_loss, digits)}<br>"
        + "Drop-out loss change: "
        + dropout_loss_change_string
        + "<br>"
        + "Variables:<br>"
        + "<br>".join(row.variables_names)
    )


def add_text_to_dendrogram(fig, updated_dendro_traces, rounding_function, digits, type="clustering"):
    res_fig = go.Figure(fig)
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
