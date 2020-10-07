from .utils import *
import plotly.express as px
import plotly.graph_objects as go
from .utils import *
from ..basics.checks import check_other_FairnessObjects
from ..._explainer.helper import verbose_cat
from ..._explainer.theme import get_default_colors
import warnings


def plot_fairness_check(fobject, title=None, other_objects=None, show=True, epsilon=0.8, verbose=True):
    data = _metric_ratios_2DF(fobject)
    n = 1
    if other_objects is not None:
        check_other_FairnessObjects(fobject, other_objects)
        for other_obj in other_objects:
            other_data = _metric_ratios_2DF(other_obj)
            data = data.append(other_data)
            n += 1

    if any(data.score == 0):
        nan_models = set(data.label[data.score == 0])
        verbose_cat(f'\nFound NaN\'s or 0\'s for models: {nan_models}\n'
                    f'It is advisable to check \'metric_ratios\'', verbose = verbose)

    upper_bound = max([max(data.score[np.invert(np.isnan(data.score.to_numpy()))]), 1 / epsilon - 1]) + 0.1
    lower_bound = min([min(data.score[np.invert(np.isnan(data.score.to_numpy()))]), epsilon - 1]) - 0.1
    lower_bound = round(lower_bound, 1)
    upper_bound = round(upper_bound, 1)

    # including upper bound
    ticks = np.arange(lower_bound, upper_bound + 0.001, step=0.1).round(1)

    # drwhy colors
    colors = get_default_colors(n, 'line')

    # change name of metrics
    data.loc[data.metric == 'TPR', 'metric'] = 'Equal opportunity ratio     TP/(TP + FN)'
    data.loc[data.metric == 'ACC', 'metric'] = 'Accuracy equality ratio    (TP + TN)/(TP + FP + TN + FN)'
    data.loc[data.metric == 'PPV', 'metric'] = 'Predictive parity ratio     TP/(TP + FP)'
    data.loc[data.metric == 'FPR', 'metric'] = 'Predictive equality ratio   FP/(FP + TN)'
    data.loc[data.metric == 'STP', 'metric'] = 'Statistical parity ratio   (TP + FP)/(TP + FP + TN + FN)'

    # make fig
    fig = px.bar(data,
                 y='subgroup',
                 x='score',
                 color='label',
                 color_discrete_sequence=colors,
                 facet_col='metric',
                 facet_col_wrap=1,
                 barmode='group',
                 orientation='h',
                 hover_name="label",
                 hover_data={
                     'label': False,
                     'metric': False,
                     'score': False,
                     'Ratio to privileged': np.round(data.score + 1, 2)}  # add 1 to scores
                 )
    # change axes range and labs
    fig.update_xaxes(tickvals=ticks,
                     ticktext=(ticks + 1).round(1),
                     range=[lower_bound, upper_bound])

    # refs are dependent on fixed numbers of metrics
    refs = ['y', 'y2', 'y3', 'y4', 'y5']
    left_red = [{'type': "rect",
                 'x0': -1,
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

    black_line = [{'type': 'line',
                   'x0': 0,
                   'x1': 0,
                   'y0': -1,
                   'y1': np.inf,
                   'yref': yref,
                   'xref': "x",
                   'line': {'color': "#371ea3", 'width': 1.5}} for yref in refs]

    for i in range(len(refs)):
        fig.add_shape(right_red[i])
        fig.add_shape(left_red[i])
        fig.add_shape(middle_green[i])
        fig.add_shape(black_line[i])

    fig.update_shapes(dict(xref='x'))

    if title is None:
        title = 'Fairness Check'

    fig.update_layout(template='plotly_white',
                      title_text =title,
                      title_x=0.5,
                      title_y = 0.99,
                      titlefont = {'size':25},
                      font={'color': "#371ea3"},
                      margin={'t': 78, 'b': 71, 'r': 30})

    # delete 'metric=' from facet names
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("metric=", ""), xanchor = 'left', x = 0, font = {'size':15}))

    # delete y axis names [fixed] number of refs
    fig['layout']['yaxis']['title']['text'] = ''
    fig['layout']['yaxis2']['title']['text'] = ''
    fig['layout']['yaxis4']['title']['text'] = ''
    fig['layout']['yaxis5']['title']['text'] = ''


    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y = 1.03,
        x=0.5,
        xanchor='center'
    ))

    return fig



def _metric_ratios_2DF(fobject):
    """
    Converts GroupFairnessClassificationObject
    to elegant DataFrame with 4 columns (subgroup, metric, score, label)
    """

    data = fobject.metric_ratios
    data = data.stack()
    data = data.reset_index()
    data.columns = ["subgroup", "metric", "score"]
    data = data.loc[data.metric.isin(fairness_check_metrics())]
    data = data.loc[data.subgroup != fobject.privileged]
    data.score -= 1

    data['label'] = np.repeat(fobject.label, data.shape[0])

    return data
