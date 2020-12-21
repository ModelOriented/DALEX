import numpy as np
import plotly.graph_objects as go


def get_default_colors(n, type):
    """return default dalex colors"""
    default_colors = ["#8bdcbe", "#f05a71", "#371ea3", "#46bac2", "#ae2c87", "#ffa58c", "#4378bf"]

    if n > len(default_colors):
        ret = []
        for _ in range(np.ceil(n / len(default_colors)).astype(int)):
            ret += default_colors
        return ret

    bar_colors = {
        1: ["#46bac2"],
        2: ["#8bdcbe", "#4378bf"],
        3: ["#8bdcbe", "#4378bf", "#46bac2"],
        4: ["#46bac2", "#371ea3", "#8bdcbe", "#4378bf"],
        5: ["#8bdcbe", "#f05a71", "#371ea3", "#46bac2", "#ffa58c"],
        6: ["#8bdcbe", "#f05a71", "#371ea3", "#46bac2", "#ae2c87", "#ffa58c"],
        7: default_colors
    }

    line_colors = {
        1: ["#46bac2"],
        2: ["#8bdcbe", "#4378bf"],
        3: ["#8bdcbe", "#f05a71", "#4378bf"],
        4: ["#8bdcbe", "#f05a71", "#4378bf", "#ffa58c"],
        5: ["#8bdcbe", "#f05a71", "#4378bf", "#ae2c87", "#ffa58c"],
        6: ["#8bdcbe", "#f05a71", "#46bac2", "#ae2c87", "#ffa58c", "#4378bf"],
        7: default_colors
    }

    if type == 'bar':
        return bar_colors[n]
    elif type == 'line':
        return line_colors[n]
    else:
        return bar_colors[n]


def get_break_down_colors():
    """return default dalex colors"""
    return ["#371ea3", "#8bdcbe", "#f05a71"]


def get_default_config():
    """return default dalex plotly config"""
    return {
                'displaylogo': False, 'staticPlot': False,
                'toImageButtonOptions': {'height': None, 'width': None, },
                'modeBarButtonsToRemove': ['sendDataToCloud', 'lasso2d', 'autoScale2d', 'select2d', 'zoom2d', 'pan2d',
                                           'zoomIn2d', 'zoomOut2d', 'resetScale2d', 'toggleSpikelines',
                                           'hoverCompareCartesian', 'hoverClosestCartesian']
            }


def fig_update_line_plot(fig, title, y_title, plot_height, hovermode):
    """update layout for CP/AP line plots"""
    for axis in fig.layout:
        if type(fig.layout[axis]) == go.layout.YAxis:  # remove redundant axis labels
            fig.layout[axis].title.text = ''
        elif type(fig.layout[axis]) == go.layout.XAxis:  # remove redundant axis labels
            fig.layout[axis].title.text = ''
        elif axis == 'annotations':
            for _, annotation in enumerate(fig.layout[axis]):  # fix annotation text
                annotation.update(text=annotation.text.split("=")[-1],
                                  font=dict(size=13))  # , x=0, xref='x' + str(index+1))  # title on the left bug

    fig.update_layout(
        # keep the original annotations and add axis title
        annotations=list(fig.layout.annotations) + [
            go.layout.Annotation(
                x=-0.07,
                y=0.5,
                font=dict(size=13),
                showarrow=False,
                text=y_title,
                textangle=-90,
                xref="paper",
                yref="paper"
            )
        ],
        font=dict(color="#371ea3"),
        margin=dict(t=78, b=71, r=30),
        hovermode=hovermode,
        title=dict(text=title, x=0.15, font=dict(size=16)),  # y=1 - 50/plot_height,
        legend=dict(
            title=dict(font=dict(size=12)),
            orientation="h",
            yanchor="bottom",
            y=1 + 30 / plot_height,
            xanchor="right",
            x=1,
            itemsizing='constant',
            font=dict(size=11)
        ),
        height=plot_height
    )

    return fig

def fig_update_bar_plot(fig, title, x_title, plot_height, hovermode):
    """update layour for CP barplot"""
    for axis in fig.layout:
        if type(fig.layout[axis]) == go.layout.YAxis:  # remove redundant axis labels
            fig.layout[axis].title.text = ''
        elif type(fig.layout[axis]) == go.layout.XAxis:  # remove redundant axis labels
            fig.layout[axis].title.text = ''
        elif axis == 'annotations':
            for _, annotation in enumerate(fig.layout[axis]):  # fix annotation text
                annotation.update(text=annotation.text.split("=")[-1],
                                  font=dict(size=13))  

    fig.update_layout(
        # keep the original annotations and add axis title
        annotations=list(fig.layout.annotations) + [
            go.layout.Annotation(
                x=0.5,
                y=-60/plot_height,
                font=dict(size=13),
                showarrow=False,
                text=x_title,
                textangle=0,
                xref="paper",
                yref="paper"
            )
        ],
        font=dict(color="#371ea3"),
        margin=dict(t=78, b=71, r=30),
        hovermode=hovermode,
        title=dict(text=title, x=0.15, font=dict(size=16)),
        legend=dict(
            title=dict(font=dict(size=12)),
            orientation="h",
            yanchor="bottom",
            y=1 + 30 / plot_height,
            xanchor="right",
            x=1,
            itemsizing='constant',
            font=dict(size=11)
        ),
        height=plot_height
    )

    return fig