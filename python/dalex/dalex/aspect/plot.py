import numpy as np
from plotly.figure_factory import create_dendrogram
import plotly.graph_objects as go
from .._global_utils import finite_range_length


def copy_dendrogram(fig):
    """Restore numeric coordinates after Plotly's typed-array serialization."""
    import base64
    result = go.Figure(fig)
    def numeric(values):
        if isinstance(values, dict) and 'bdata' in values and 'dtype' in values:
            return np.frombuffer(base64.b64decode(values['bdata']), dtype=values['dtype']).copy()
        return values
    for trace in result.data:
        trace.x = numeric(trace.x)
        trace.y = numeric(trace.y)
    result.layout.yaxis.tickvals = numeric(result.layout.yaxis.tickvals)
    return result

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

    fig.update_layout(
        xaxis={
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "ticks": "",
            "fixedrange": True,
        },
        yaxis={
            "mirror": False,
            "showgrid": False,
            "showline": False,
            "zeroline": False,
            "ticks": "",
            "fixedrange": True,
        },
        font={"color": "#371ea3"}, 
        template="none", 
        margin={"t": 78, "b": 71, "r": 30}
    )

    return fig


def add_text_and_tooltips_to_dendrogram(fig, _dendrogram_aspects_ordered, rounding_function, digits):
    from .._explanation import IndexedTable
    from ._plotting import as_plot_data
    source = _dendrogram_aspects_ordered
    if isinstance(source, IndexedTable):
        source = source.data
    data = as_plot_data(source)
    rows = data.rows()
    # Pandas scalar callbacks keep their historical scalar types. Native
    # backends need only the small numerical column, not a pandas publication.
    numbers = data.frame['min_depend'].to_native() if data.pandas_rounding else data.frame['min_depend'].to_numpy()
    res_fig = copy_dendrogram(fig)
    corner_scatters = [] # scatters with depend=0
    for i, scatter in enumerate(res_fig.data):
        # 0 ---- 1
        #        | 
        #        *   scatter indexing convention
        #        |
        # 3 ---- 2
        # add middle point - for text label (*) 
        x_cord = scatter.x[1] 
        y_cord = np.mean(scatter.y[1:3])
        scatter.x = np.insert(scatter.x, 2, x_cord)   
        scatter.y = np.insert(scatter.y, 2, y_cord)

        # add text labels and tooltips
        minimum = numbers.iloc[i] if data.pandas_rounding else numbers[i]
        min_depend_val_rounded = str(rounding_function(minimum, digits))
        if x_cord != 1:
            scatter.mode = "text+lines"
            scatter["text"] = [None, None, str(rounding_function(1-x_cord, digits)), None, None]
            scatter["textposition"] = "middle left"
            scatter.hoverinfo = "text"
            scatter.hoverlabel = {"bgcolor": "rgba(0,0,0,0.8)"}
            scatter.hovertext = (
                "Min depend value: " + min_depend_val_rounded
                + "<br>(between variables: " + ", ".join(rows[i]['vars_min_depend']) + ")"
                + "<br>Variables:<br>" + "<br>".join(rows[i]['variable_names'])
            )
        else:
            scatter.hoverlabel = {"bgcolor": "rgba(0,0,0,0.8)"}
            scatter.hovertext = (
                "Min depend value: " + min_depend_val_rounded
                +  "<br>(between variables: " + ", ".join(rows[-1]['vars_min_depend']) + ")"
                + "<br>Variables:<br>" + "<br>".join(rows[-1]['variable_names'])
            )
            corner_scatters.append(scatter)

    if corner_scatters:  
        # (add only one text label for all)
        scatter = corner_scatters[len(corner_scatters)//2]
        label = str(rounding_function(0, digits))
        scatter.mode = "text+lines"
        scatter["text"] = [None, None, label, None, None]
        scatter["textposition"] = "middle left"
        scatter.hoverinfo = "text"
            

    return res_fig


def _add_between_points(
    arr, arr_2, val_arr_2, range_global, i_begin, i_end, arr_text=None
):
    # helper function to add more points to trace (dendrogram line)
    # number of added points is based on length of line (min 5)
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
    # add points to all traces in dendrogram 
    k = len(fig.data)
    # Hover interpolation needs coordinate spans, not a browser-rendered
    # figure. This also handles constant panels without a zero denominator.
    range_x_len = finite_range_length(fig.layout.xaxis.range or [
        value for trace in fig.data if trace.x is not None for value in trace.x])
    range_y_len = finite_range_length(fig.layout.yaxis.range or [
        value for trace in fig.data if trace.y is not None for value in trace.y])
    for i in range(k):
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
    return fig
