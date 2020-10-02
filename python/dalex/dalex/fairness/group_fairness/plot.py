import numpy as np

def plot_fairness_check(fobject, other_objects, epsilon = 0.8, **kwargs):

    data = fobject.metric_ratios

    data = data.stack()
    data = data.reset_index()
    data.columns = ["subgroup", "metric", "score"]
    data = data.loc[data.metric.isin(['ACC', 'TPR','FPR', 'PPV', 'STP'])]
    data = data.loc[data.subgroup != fobject.privileged]
    data.score -= 1

    upper_bound = max([max(data.score[np.invert(np.isnan(data.score.to_numpy()))]), 1 / epsilon - 1]) + 0.1
    lower_bound = min([min(data.score[np.invert(np.isnan(data.score.to_numpy()))]), epsilon - 1]) - 0.1
    lower_bound = round(lower_bound, 1)
    upper_bound = round(upper_bound, 1)

    ticks = np.arange(lower_bound, upper_bound, step=0.2)

    if any(np.isin(ticks, 0)):
        ticks += 0.1
        ticks = ticks.round(1)

    import plotly.express as px

    fig = (px.bar(data, y='subgroup', x='score', facet_row='metric', orientation='h'))

    fig.update_xaxes(tickvals=ticks,
                     ticktext=(ticks + 1).round(1),
                     range=[lower_bound, upper_bound])

    refs = ['y', 'y2', 'y3', 'y4','y5']
    left_red = [{'type': "rect",
                 'x0': lower_bound,
                 'y0': -1,
                 'x1': epsilon - 1,
                 'y1': np.inf,
                 'yref': yref,
                 'line': {'width': 0},
                 'fillcolor': '#f05a71',
                 'layer': 'below',
                 'opacity': 0.1} for yref in refs]

    middle_green = [{'type': "rect",
                     'x0': epsilon - 1,
                     'y0': -1,
                     'x1': 1 / epsilon - 1,
                     'y1': np.inf,
                     'yref': yref,
                     'line': {'width': 0},
                     'fillcolor': '#c7f5bf',
                     'layer': 'below',
                     'opacity': 0.1} for yref in refs]

    right_red = [{'type': "rect",
                  'x0': 1 / epsilon - 1,
                  'y0': -1,
                  'x1': 100000,
                  'y1': np.inf,
                  'yref': yref,
                  'line': {'width': 0},
                  'fillcolor': '#f05a71',
                  'layer': 'below',
                  'opacity': 0.1} for yref in refs]

    for i in range(len(refs)):
        fig.add_shape(right_red[i])
        fig.add_shape(left_red[i])
        fig.add_shape(middle_green[i])

    fig.update_shapes(dict(xref='x'))
    fig.update_layout(template='plotly_white')
    fig.show()







