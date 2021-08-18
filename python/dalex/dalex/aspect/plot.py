from plotly.figure_factory import create_dendrogram
import plotly.graph_objects as go
import numpy as np

def plot_dendrogram(linkage_matrix, labels=None):
    fig = create_dendrogram(
        linkage_matrix,
        labels=labels,
        distfun=lambda x: x,
        linkagefun=lambda x: x,
        orientation="left",
        color_threshold=-0.1,
        colorscale=["#46bac2"],
    )
    if labels is None:
        fig.update_yaxes(visible=False, showticklabels=False)
    
    fig.update_xaxes(range=[-0.05, 1.05], visible=False, showticklabels=False)

    # Edit xaxis
    fig.update_layout(
        xaxis={
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "ticks": "",
            "fixedrange": True,
        }
    )

    # Edit yaxis
    fig.update_layout(
        yaxis={
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "ticks": "",
            "fixedrange": True,
        }
    )

    fig.update_layout(
        font={"color": "#371ea3"}, template="none", margin={"t": 78, "b": 71, "r": 30}
    )

    return fig


def add_text_and_tooltips_to_dendrogram(fig, _dendrogram_aspects_ordered, rounding_function, digits):
    res_fig = go.Figure(fig)
    corner_scatters = []
    for i, scatter in enumerate(res_fig.data):
        x_cord = scatter.x[1]
        y_cord = np.mean(scatter.y[1:3])
        scatter.x = np.insert(scatter.x, 2, x_cord)
        scatter.y = np.insert(scatter.y, 2, y_cord)
        min_depend_val_rounded = str(rounding_function(_dendrogram_aspects_ordered.iloc[i].min_depend, digits))
        if x_cord != 1:
            label = "{0:.2f}".format(1 - x_cord)
            scatter.mode = "text+lines"
            scatter["text"] = [None, None, label, None, None]
            scatter["textposition"] = "middle left"
            scatter.hoverinfo = "text"
            scatter.hoverlabel = {"bgcolor": "rgba(0,0,0,0.8)"}
            scatter.hovertext = (
                "Min depend value: " + min_depend_val_rounded
                + "<br>(between variables: " + ", ".join(_dendrogram_aspects_ordered.iloc[i].vars_min_depend) + ")"
                + "<br>Variables:<br>" + "<br>".join(_dendrogram_aspects_ordered.iloc[i].variables_names)
            )
        else:
            scatter.hoverlabel = {"bgcolor": "rgba(0,0,0,0.8)"}
            scatter.hovertext = (
                "Min depend value: " + min_depend_val_rounded
                +  "<br>(between variables: " + ", ".join(_dendrogram_aspects_ordered.iloc[-1].vars_min_depend) + ")"
                + "<br>Variables:<br>" + "<br>".join(_dendrogram_aspects_ordered.iloc[-1].variables_names)
            )
            corner_scatters.append(scatter)

    if corner_scatters:  
        scatter = corner_scatters[len(corner_scatters)//2]
        label = "0.00"
        scatter.mode = "text+lines"
        scatter["text"] = [None, None, label, None, None]
        scatter["textposition"] = "middle left"
        scatter.hoverinfo = "text"
            

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
    range_x = fig.full_figure_for_development(warn=False).layout.xaxis.range
    range_x_len = range_x[1] - range_x[0]
    range_y = fig.full_figure_for_development(warn=False).layout.yaxis.range
    range_y_len = range_y[1] - range_y[0]
    for i in range(k):
        arr_text = None
        num_points = len(fig.data[i]["x"])
        if num_points == 5:
            arr_text = fig.data[i]["text"]
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
    return fig