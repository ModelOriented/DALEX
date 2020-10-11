from .utils import *
import plotly.express as px
import plotly.graph_objects as go
from .utils import *
from ..basics.checks import check_other_FairnessObjects
from ..._explainer.helper import verbose_cat
from ..._explainer.theme import get_default_colors
from plotly.validators.scatter.marker import SymbolValidator
import warnings


def plot_fairness_check(fobject,
                        title=None,
                        other_objects=None,
                        epsilon=0.8,
                        verbose=True):
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
                    f'It is advisable to check \'metric_ratios\'', verbose=verbose)

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

    # without privileged
    data = data.loc[data.subgroup != fobject.privileged]

    # subgroup y-axis value creation
    n_ticks = len(data.subgroup.unique())
    tick_values = np.linspace(0, 1, n_ticks + 2)[1:-1]

    subgroup_tick_dict = {}
    for i in range(len(data.subgroup.unique())):
        subgroup = data.subgroup.unique()[i]
        subgroup_tick_dict[subgroup] = tick_values[i]

    data['subgroup_numeric'] = [subgroup_tick_dict.get(sub) for sub in data.subgroup]
    data = data.reset_index(drop=True)

    # for hover
    data['dispx'] = np.round(data.score + 1, 3)
    # make fig
    fig = px.bar(data,
                 y='subgroup_numeric',
                 x='score',
                 color='label',
                 color_discrete_sequence=colors,
                 facet_col='metric',
                 facet_col_wrap=1,
                 barmode='group',
                 orientation='h',
                 custom_data=['subgroup', 'dispx', 'label']
                 )

    fig.update_traces(
        hovertemplate="<br>".join([
            "<b>%{customdata[2]}</b><br>"
            "Subgroup: %{customdata[0]}",
            "Score: %{customdata[1]}",
            "<extra></extra>"

        ]))
    fig.update_layout(hoverlabel_align = 'left')

    fig.update_yaxes(tickvals=list(subgroup_tick_dict.values()),
                     ticktext=list(subgroup_tick_dict.keys()))

    # change axes range and labs
    fig.update_xaxes(tickvals=ticks,
                     ticktext=(ticks + 1).round(1),
                     range=[lower_bound, upper_bound])

    # refs are dependent on fixed numbers of metrics
    refs = ['y', 'y2', 'y3', 'y4', 'y5']
    left_red = [{'type': "rect",
                 'x0': -1,
                 'y0': 0,
                 'x1': epsilon - 1,
                 'y1': 1,
                 'yref': yref,
                 'line': {'width': 0},
                 'fillcolor': '#f05a71',
                 'layer': 'below',
                 'opacity': 0.1} for yref in refs]

    middle_green = [{'type': "rect",
                     'x0': epsilon - 1,
                     'y0': 0,
                     'x1': 1 / epsilon - 1,
                     'y1': 1,
                     'yref': yref,
                     'line': {'width': 0},
                     'fillcolor': '#c7f5bf',
                     'layer': 'below',
                     'opacity': 0.1} for yref in refs]

    right_red = [{'type': "rect",
                  'x0': 1 / epsilon - 1,
                  'y0': 0,
                  'x1': 100000,  # hard coded :(
                  'y1': 1,
                  'yref': yref,
                  'line': {'width': 0},
                  'fillcolor': '#f05a71',
                  'layer': 'below',
                  'opacity': 0.1} for yref in refs]

    black_line = [{'type': 'line',
                   'x0': 0,
                   'x1': 0,
                   'y0': 0,
                   'y1': 1,
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
                      title_text=title,
                      title_x=0.5,
                      title_y=0.99,
                      titlefont={'size': 25},
                      font={'color': "#371ea3"},
                      margin={'t': 78, 'b': 71, 'r': 30})

    # delete 'metric=' from facet names
    fig.for_each_annotation(
        lambda a: a.update(text=a.text.replace("metric=", ""), xanchor='left', x=0.05, font={'size': 15}))

    # delete y axis names [fixed] number of refs
    for i in ['', '2', '4', '5']:
        fig.update_layout({'yaxis' + i + '_title_text': ''})

    return fig


def plot_metric_scores(fobject,
                       other_objects,
                       title=None):
    data = fobject.subgroup_metrics.to_vertical_DataFrame()
    data['label'] = np.repeat(fobject.label, data.shape[0]).astype('U')
    n = 1
    if other_objects is not None:
        check_other_FairnessObjects(fobject, other_objects)
        for other_obj in other_objects:
            other_data = other_obj.subgroup_metrics.to_vertical_DataFrame()
            other_data['label'] = np.repeat(other_obj.label, other_data.shape[0]).astype('U')
            data = data.append(other_data)
            n += 1

    # metric choosing and name change
    data = data.loc[data.metric.isin(fairness_check_metrics())]
    data.loc[data.metric == 'TPR', 'metric'] = 'TPR    TP/(TP + FN)'
    data.loc[data.metric == 'ACC', 'metric'] = 'ACC   (TP + TN)/(TP + FP + TN + FN)'
    data.loc[data.metric == 'PPV', 'metric'] = 'PPV    TP/(TP + FP)'
    data.loc[data.metric == 'FPR', 'metric'] = 'FPR    FP/(FP + TN)'
    data.loc[data.metric == 'STP', 'metric'] = 'STP   (TP + FP)/(TP + FP + TN + FN)'

    # for x axis
    min_score = min(data.score)
    max_score = max(data.score)

    # subgroup y-axis value creation
    n_ticks = len(data.label.unique())
    tick_values = np.linspace(0, 1, n_ticks + 2)[1:-1]

    privileged_data = data.loc[data.subgroup == fobject.privileged]
    data = data.loc[data.subgroup != fobject.privileged]

    subgroup_tick_dict = {}
    label_tick_dict = {}
    for i in range(len(data.subgroup.unique())):
        subgroup = data.subgroup.unique()[i]
        subgroup_tick_dict[subgroup] = i

    for i in range(len(data.label.unique())):
        label = data.label.unique()[i]
        label_tick_dict[label] = tick_values[i]

    data['subgroup_numeric'] = [subgroup_tick_dict.get(sub) for sub in data.subgroup]
    data = data.reset_index(drop=True)
    data.subgroup_numeric = data.subgroup_numeric + pd.Series([label_tick_dict.get(lab) for lab in data.label])

    # drwhy colors
    colors = get_default_colors(len(data.label.unique()), 'line')

    # fig creation
    fig = px.scatter(data,
                     x='score',
                     y='subgroup_numeric',
                     color='label',
                     color_discrete_sequence=colors,
                     facet_col='metric',
                     facet_col_wrap=1,
                     custom_data=['subgroup'])

    fig.update_traces(
        hovertemplate="<br>".join([
            "Subgroup: %{customdata[0]}",
            "Score: %{x}"
        ]),
        hoverlabel={'align': np.where(data.score > 0, 'right', 'left')}
    )

    fig.update_traces(mode='markers',
                      marker_size=10)

    fig.update_xaxes(tickvals=np.arange(0, 1.01, 0.1))

    # cols and ref dicts are dependent on the arrangement of metrics and labels
    color_dict = {}
    k = 0
    for label in data.label.unique():
        color_dict[label] = colors[k]
        k += 1

    refs = ['y', 'y2', 'y3', 'y4', 'y5']
    refs_dict = {}
    j = 1
    for metric in data.metric.unique():
        refs_dict[metric] = refs[len(refs) - j]
        j += 1
    # add lines
    for metric in data.metric.unique():
        for label in data.label.unique():
            x = float(privileged_data.loc[(privileged_data.metric == metric) &
                                          (privileged_data.label == label), :].score)
            # lines
            for subgroup in data.subgroup.unique():
                y = float(data.loc[(data.metric == metric) &
                                   (data.label == label) &
                                   (data.subgroup == subgroup)].subgroup_numeric)
                # horizontal
                fig.add_shape(type='line',
                              xref='x',
                              yref=refs_dict.get(metric),
                              x0=float(data.loc[(data.metric == metric) &
                                                (data.label == label) &
                                                (data.subgroup == subgroup)].score),
                              x1=x,
                              y0=y,
                              y1=y,
                              line=dict(
                                  color=color_dict.get(label),
                                  width=1),
                              layer='below')
            # vertical
            fig.add_shape(type='line',
                          xref='x',
                          yref=refs_dict.get(metric),
                          x0=x,
                          x1=x,
                          y0=0,
                          y1=np.ceil(max(data.subgroup_numeric)),
                          line=dict(
                              color=color_dict.get(label),
                              width=2),
                          layer='below')

    if title is None:
        title = 'Metric Scores'

    fig.update_layout(title_text=title,
                      template='plotly_white',
                      title_x=0.5,
                      title_y=0.99,
                      titlefont={'size': 25},
                      font={'color': "#371ea3"},
                      margin={'t': 78, 'b': 71, 'r': 30})

    fig.for_each_annotation(
        lambda a: a.update(text=a.text.replace("metric=", ""), xanchor='left', x=0.05, font={'size': 15}))

    # fig.update_layout(legend=dict(
    #     orientation="h",
    #     yanchor="bottom",
    #     y=1.03,
    #     x=0.5,
    #     xanchor='center'
    # ))

    # disable all y grids and
    # delete y axis names [fixed] number of refs
    for i in ['', '2', '3', '4', '5']:
        fig.update_layout({'yaxis' + i + '_showgrid': False,
                           'yaxis' + i + '_zeroline': False})
        if i != '3':
            fig.update_layout({'yaxis' + i + '_title_text': ''})

    # names on y axis - each subgroup in middle of integers - 0.5, 1.5 ,...
    subgroup_tick_dict_updated = {}
    for sub, val in subgroup_tick_dict.items():
        subgroup_tick_dict_updated[sub] = val + 0.5

    # centers axis values
    fig.update_yaxes(tickvals=list(subgroup_tick_dict_updated.values()),
                     ticktext=list(subgroup_tick_dict_updated.keys()))

    # fixes rare bug where axis are in center and blank fields on left and right
    fig.update_xaxes(range=[min_score - 0.05, max_score + 0.05])

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
