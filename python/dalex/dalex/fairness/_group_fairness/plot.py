import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import utils
from .._basics import checks as basic_checks
from .._basics.exceptions import ParameterCheckError
from ..._explainer import helper
from ... import _theme


def plot_fairness_check(fobject,
                        title=None,
                        other_objects=None,
                        epsilon=0.8,
                        verbose=True):
    data = utils._metric_ratios_2_df(fobject)
    n = 1
    if other_objects is not None:
        basic_checks.check_other_fairness_objects(fobject, other_objects)
        for other_obj in other_objects:
            other_data = utils._metric_ratios_2_df(other_obj)
            data = data.append(other_data)
            n += 1

    if any(data.score == 0):
        nan_models = set(data.label[data.score == 0])
        helper.verbose_cat(f'\nFound NaN\'s or 0\'s for models: {nan_models}\n'
                           f'It is advisable to check \'metric_ratios\'', verbose=verbose)

    upper_bound = max([max(data.score[np.invert(np.isnan(data.score.to_numpy()))]), 1 / epsilon - 1]) + 0.1
    lower_bound = min([min(data.score[np.invert(np.isnan(data.score.to_numpy()))]), epsilon - 1]) - 0.1
    lower_bound = round(lower_bound, 1)
    upper_bound = round(upper_bound, 1)

    # including upper bound
    ticks = np.arange(lower_bound, upper_bound + 0.001, step=0.1).round(1)

    # drwhy colors
    colors = _theme.get_default_colors(n, 'line')

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

    # change axes range and labs
    fig.update_xaxes(tickvals=ticks,
                     ticktext=(ticks + 1).round(1),
                     range=[lower_bound, upper_bound])

    fig.update_yaxes(tickvals=list(subgroup_tick_dict.values()),
                     ticktext=list(subgroup_tick_dict.keys()),
                     range=[0, 1])

    # refs are dependent on fixed numbers of metrics
    refs = ['y', 'y2', 'y3', 'y4', 'y5']
    left_red = [{'type': "rect",
                 'x0': lower_bound,
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
                  'x1': upper_bound,
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

    fig.update_layout(utils._fairness_theme(title))

    # delete 'metric=' from facet names
    fig.for_each_annotation(
        lambda a: a.update(text=a.text.replace("metric=", ""), xanchor='left', x=0.05, font={'size': 15}))

    # delete y axis names [fixed] number of refs
    for i in ['', '2', '4', '5']:
        fig.update_layout({'yaxis' + i + '_title_text': ''})

    fig.update_layout({'yaxis3_title_text': 'subgroup'})
    fig.update_yaxes(showgrid=False, zeroline=False)

    return fig


def plot_metric_scores(fobject,
                       other_objects,
                       title=None):
    data = fobject._subgroup_confusion_matrix_metrics_object.to_vertical_DataFrame()
    data['label'] = np.repeat(fobject.label, data.shape[0]).astype('U')
    n = 1
    if other_objects is not None:
        basic_checks.check_other_fairness_objects(fobject, other_objects)
        for other_obj in other_objects:
            other_data = other_obj._subgroup_confusion_matrix_metrics_object.to_vertical_DataFrame()
            other_data['label'] = np.repeat(other_obj.label, other_data.shape[0]).astype('U')
            data = data.append(other_data)
            n += 1

    # metric choosing and name change
    data = data.loc[data.metric.isin(["TPR", "PPV", "STP", "ACC", "FPR"])]
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
    colors = _theme.get_default_colors(len(data.label.unique()), 'line')

    # fig creation
    fig = px.scatter(data,
                     x='score',
                     y='subgroup_numeric',
                     color='label',
                     color_discrete_sequence=colors,
                     facet_col='metric',
                     facet_col_wrap=1,
                     custom_data=['subgroup', 'label'])

    fig.update_traces(
        hovertemplate="<br>".join([
            "<b>%{customdata[1]}</b><br>"
            "Subgroup: %{customdata[0]}",
            "Score: %{x}"
            "<extra></extra>"

        ]))

    # point size
    fig.update_traces(mode='markers',
                      marker_size=7)

    # axis ticks
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
                                  width=1))
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
                              width=2))

    # theme and appearance
    if title is None:
        title = 'Metric Scores'

    fig.update_layout(utils._fairness_theme(title))

    fig.for_each_annotation(
        lambda a: a.update(text=a.text.replace("metric=", ""), xanchor='left', x=0.05, font={'size': 15}))

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

    # delete y axis names [fixed] number of refs
    for i in ['', '2', '4', '5']:
        fig.update_layout({'yaxis' + i + '_title_text': ''})

    fig.update_layout({'yaxis3_title_text': 'subgroup'})

    # fixes rare bug where axis are in center and blank fields on left and right
    fig.update_xaxes(range=[min_score - 0.05, max_score + 0.05])
    fig.update_yaxes(showgrid=False, zeroline=False)

    return fig


def plot_stacked(fobject,
                 title=None,
                 other_objects=None,
                 metrics=["TPR", "PPV", "STP", "ACC", "FPR"],
                 verbose=True,
                 **kwargs):
    data = utils._unwrap_parity_loss_data(fobject, other_objects, metrics, verbose)

    fig = px.bar(data,
                 x='score',
                 y='label',
                 custom_data=[np.round(data.score, 3), data.metric],
                 labels={'score': 'cumulated parity loss'},
                 orientation='h',
                 color='metric',
                 color_discrete_sequence=_theme.get_default_colors(len(metrics), type='line'))
    # no outline
    fig.update_traces(marker_line_width=0)

    # hover

    if title is None:
        title = "Stacked Parity Loss Metrics"
    fig.update_layout(utils._fairness_theme(title))
    fig.update_yaxes(showgrid=False, zeroline=False)

    fig.update_traces(
        hovertemplate="<br>".join([
            "<b>%{label}</b><br>"
            "Metric: %{customdata[1]}<br>"
            "Parity loss: %{customdata[0]}",
            "<extra></extra>"
        ]))

    return fig


def plot_radar(fobject,
               other_objects=None,
               title=None,
               verbose=True,
               metrics=["TPR", "ACC", "PPV", "FPR", "STP"],
               **kwargs):
    data = utils._unwrap_parity_loss_data(fobject, other_objects, metrics, verbose)
    colors = _theme.get_default_colors(len(set(data.label)), type='line')

    fig = go.Figure()
    for i, label in enumerate(set(sorted(data.label))):
        model_data = data.loc[data.label == label, :]
        r = list(model_data.score)
        r.append(r[0])
        theta = list(model_data.metric)
        theta.append(theta[0])
        fig.add_trace(
            go.Scatterpolar(
                r=r,
                theta=theta,
                name=label,
                marker=dict(color=[colors[i] for elem in r]),
                line=dict(color=colors[i]),
                text=[label for _ in r]
            )
        )

    if title is None:
        title = "Fairness Radar"
    fig.update_layout(utils._fairness_theme(title))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                range=[0, max(data.score) + 0.1]
            ))
    )

    fig.update_traces(
        hovertemplate="<br>".join([
            "<b>%{text}</b><br>"
            "Metric: %{theta}<br>"
            "Parity loss: %{r:.3f}"
            "<extra></extra>"
        ]))

    return fig


def plot_performance_and_fairness(fobject,
                                  other_objects=None,
                                  fairness_metric='TPR',
                                  performance_metric='accuracy',
                                  title=None,
                                  verbose=True,
                                  **kwargs):
    data = utils._unwrap_parity_loss_data(fobject, other_objects, [fairness_metric], verbose)
    assert len(data.label.unique()) == len(data.label)

    performance_data = pd.DataFrame(columns=['label', 'performance_score'])
    performance_data.loc[0] = [fobject.label, utils._classification_performance(fobject, verbose, performance_metric)]

    if other_objects:
        for i, obj in enumerate(other_objects):
            performance_data.loc[i + 1] = [obj.label,
                                           utils._classification_performance(obj, verbose, performance_metric)]

    data = data.merge(performance_data, on='label')

    fig = px.scatter(data,
                     x='performance_score',
                     y='score',
                     color='label',
                     custom_data=[data.label],
                     color_discrete_sequence=_theme.get_default_colors(len(data.label), 'line'))

    fig.update_traces(
        mode='markers',
        marker=dict(size=[15 for _ in data.label]))

    if title is None:
        title = "Performance and Fairness"
    fig.update_layout(utils._fairness_theme(title))

    fig.update_yaxes(title="reversed " + fairness_metric + " parity loss",
                     autorange='reversed')
    fig.update_xaxes(title=performance_metric)

    fig.update_traces(
        hovertemplate="<br>".join([
            "<b>%{customdata[0]}</b><br>",
            fairness_metric + " parity loss: %{y:.3f}",
            performance_metric + " score: %{x:.3f}",
            "<extra></extra>"
        ]))

    return fig


def plot_heatmap(fobject,
                 other_objects=None,
                 metrics='all',
                 title=None,
                 verbose=True,
                 **kwargs):
    if metrics == 'all':
        metrics = list(fobject.parity_loss.index)

    data = utils._unwrap_parity_loss_data(fobject, other_objects, metrics, verbose=verbose)
    score_data = data.score.values.reshape(len(data.label.unique()), len(data.metric.unique()))

    fig = px.imshow(score_data,
                    labels=dict(x="parity loss metrics", y="model", color="Metric's parity loss"),
                    x=list(data.metric.unique()),
                    y=list(data.label.unique()),
                    color_continuous_scale=["#c7f5bf", "#8bdcbe", "#46bac2", "#4378bf", "#371ea3"])

    if title is None:
        title = "Fairness Heatmap"
    fig.update_layout(utils._fairness_theme(title))

    fig.update_traces(
        hovertemplate="<br>".join([
            "<b>%{y}</b><br>",
            "Parity loss metric: %{x}",
            "parity loss: %{z:.3f}",
            "<extra></extra>"
        ]))

    return fig


def plot_ceteris_paribus_cutoff(fobject,
                                other_objects=None,
                                title=None,
                                verbose=True,
                                grid_points=101,
                                subgroup=None,
                                metrics=["TPR", "ACC", "PPV", "FPR", "STP"],
                                **kwargs):
    if subgroup is None:
        raise ParameterCheckError("parameter \'subgroup\' is needed")

    protected = fobject.protected
    privileged = fobject.privileged
    if subgroup not in protected:
        raise ParameterCheckError("parameter subgroup must be in protected parameter")

    objects = [fobject]
    if other_objects is not None:
        basic_checks.check_other_fairness_objects(fobject, other_objects)
        for obj in other_objects:
            objects.append(obj)

    colors = _theme.get_default_colors(len(metrics), 'line')

    labels = []
    for obj in objects:
        labels.append(obj.label)

    fig = make_subplots(rows=len(objects), cols=1, subplot_titles=labels)

    for k, object in enumerate(objects):
        cutoff = object.cutoff
        y = object.y
        y_hat = object.y_hat
        data = pd.DataFrame()

        # generate data for hypothetical cutoffs
        for i in range(1, grid_points):
            cutoff[subgroup] = i / grid_points
            sub_confusion_matrix = utils.SubgroupConfusionMatrix(y_true=y,
                                                                 y_pred=y_hat,
                                                                 protected=protected,
                                                                 cutoff=cutoff)

            sub_confusion_matrix_metrics = utils.SubgroupConfusionMatrixMetrics(sub_confusion_matrix)
            parity_loss = utils.calculate_parity_loss(sub_confusion_matrix_metrics, privileged)
            parity_loss = parity_loss.loc[parity_loss.index.isin(metrics)]
            newdata = pd.DataFrame({'score': parity_loss, 'metric': parity_loss.index, 'cutoff': i / (grid_points - 1)})
            newdata = newdata.reset_index(drop=True)
            data = data.append(newdata)

        data = data.reset_index(drop=True)

        # Find minimum where NA is not present
        pivoted_data = data.pivot(index='cutoff', columns='metric', values='score')
        summed_metrics = pivoted_data.to_numpy().sum(axis=1)
        min_index = np.where(summed_metrics == min(summed_metrics))[0][0]  # get first minimal index
        min_cutoff = data.cutoff.unique()[min_index]

        # make figure from individual parts
        for j, metric in enumerate(metrics):
            metric_data = data.loc[data.metric == metric, :]
            fig.add_trace(go.Scatter(x=metric_data.cutoff,
                                     y=metric_data.score,
                                     mode='lines',
                                     line=dict(color=colors[j]),
                                     name=metric,
                                     showlegend=k == 0),  # show only legend of first subplot
                          row=k + 1, col=1)

        fig.add_shape(type='line',
                      x0=min_cutoff,
                      x1=min_cutoff,
                      y0=0,
                      y1=max(data.score),
                      layer='below',
                      line=dict(color="grey",
                                dash="dot"), row=k + 1, col=1)

        fig.add_annotation(
            x=min_cutoff,
            y=max(data.score) + 0.05,
            text=f'minimum: {min_cutoff}',
            row=k + 1, col=1
        )

    fig.update_xaxes(title=f"cutoff for {subgroup}, other cutoffs constant")
    fig.update_yaxes(title="parity loss")

    if title is None:
        title = "Ceteris Paribus Cutoff"
    fig.update_layout(utils._fairness_theme(title))
    fig.update_layout(hovermode="x")
    fig.update_traces(
        hovertemplate="<br>".join([
            "parity loss: %{y:.3f}"
        ]))

    return fig


def plot_boxplot(fobject,
                 other_objects,
                 title,
                 show):
    data = pd.DataFrame(columns=['y', 'y_hat', 'subgroup', 'model'])
    objects = [fobject]
    if other_objects is not None:
        for other_obj in other_objects:
            objects.append(other_obj)
    for obj in objects:
        for subgroup in obj.regression_dict.regression_dict.keys():
            y, y_hat = obj.regression_dict.regression_dict[subgroup].values()
            data_to_append = pd.DataFrame({'y': y,
                                           'y_hat': y_hat,
                                           'subgroup': np.repeat(subgroup, len(y)),
                                           'model': np.repeat(obj.label, len(y))})
            data = data.append(data_to_append)

    fig = go.Figure()

    counter = 0
    for model in data.model.unique():
        for i, sub in enumerate(data.subgroup.unique()):
            counter += 1
            fig.add_trace(
                go.Violin(
                    box_visible = True,
                    x = data.loc[(data.subgroup == sub) & (data.model == model)].y_hat,
                    y0 = sub + model,
                    name = sub,
                    fillcolor = _theme.get_default_colors(len(data.subgroup.unique()), type='line')[i],
                    opacity=0.9,
                    line_color = 'black'
                )
            )

    violins_in_model =  int(counter/len(data.model.unique()))
    starter_violins = np.arange(0, counter, violins_in_model)

    fig.update_xaxes(title='prediction')
    fig.update_yaxes(title='model', tickvals= list((starter_violins + (violins_in_model-1)/2)), ticktext=list(data.model.unique()))

    # hide doubling entries in legend
    legend_entries = set()
    for trace in fig['data']:
        legend_entries.add(trace['name'])

    for trace in fig['data']:
        if trace['name'] in legend_entries:
            legend_entries.remove(trace['name'])
        else:
            trace['showlegend'] = False

    if title is None:
        title = "Density plot"
    fig.update_layout(utils._fairness_theme(title))

    return fig
