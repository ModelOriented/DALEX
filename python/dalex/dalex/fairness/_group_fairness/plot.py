import numpy as np
import narwhals.stable.v2 as nw
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import checks
from . import utils
from .._basics import checks as basic_checks
from .._basics.exceptions import ParameterCheckError
from ... import _theme
from ..._explainer import helper
from ..._plotting import concat_frames
from ._plotting import table_frame, long_table, parity_data, finite_bounds, unique, rename_metrics, score_matrix, first


def plot_fairness_check(data, n_models, epsilon, title):
    if not len(data):
        raise ParameterCheckError('No unprivileged subgroup scores are available to plot')
    minimum, maximum = finite_bounds(data['score'].to_numpy(), default=(0., 0.))
    upper_bound = max([maximum, 1 / epsilon - 1])
    upper_bound = upper_bound * 1.05
    lower_bound = min([minimum, epsilon - 1])
    lower_bound = lower_bound * 1.1
    upper_bound = round(upper_bound, 1)

    # drwhy colors
    colors = _theme.get_default_colors(n_models, 'line')

    # subgroup y-axis value creation
    subgroups = unique(data, 'subgroup')
    n_ticks = len(subgroups)
    tick_values = np.linspace(0, 1, n_ticks + 2)[1:-1]

    subgroup_tick_dict = {}
    for i in range(len(subgroups)):
        subgroup = subgroups[i]
        subgroup_tick_dict[subgroup] = tick_values[i]

    data = data.with_columns(nw.new_series('subgroup_numeric',
        [subgroup_tick_dict.get(sub) for sub in data['subgroup'].to_list()], backend=nw.get_native_namespace(data)))

    # for hover
    data = data.with_columns(nw.new_series('dispx', np.round(data['score'].to_numpy().astype(float) + 1, 3),
                                          backend=nw.get_native_namespace(data)))
    # make fig
    fig = px.bar(data.to_native(),
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
    # including upper bound
    # if upper_bound < 1:
    #     # tick are calculated by backend data
    #     if upper_bound < 0.4 and lower_bound > -0.4:
    #         ticks = np.arange(lower_bound, upper_bound + 0.001, step=0.1).round(1)
    #     else:
    #         ticks = np.arange(lower_bound, upper_bound + 0.001, step=0.2).round(1)
    #
    #     ticks = np.append(ticks, 0)
    #     fig.update_xaxes(tickvals=ticks,
    #                      ticktext=(ticks + 1).round(1),
    #                      range=[lower_bound, upper_bound])
    #
    # else:
    # ticks are calculated by appearance
    min_value, max_value, spacing = utils.get_nice_ticks(lower_bound+1, upper_bound+1)

    ticks = np.arange(min_value, max_value, step=spacing).round(1)
    ticks = np.append(ticks, 1)
    fig.update_xaxes(tickvals=ticks-1,
                     ticktext=ticks,
                     range=[lower_bound, upper_bound])

    fig.update_yaxes(tickvals=list(subgroup_tick_dict.values()),
                     ticktext=list(subgroup_tick_dict.keys()),
                     range=[0, 1])

    # refs are dependent on fixed numbers of metrics

    n_metrics = len(unique(data, 'metric'))
    refs = ['y' + str(i) for i in range(2, n_metrics + 1)]
    refs.insert(0, 'y')

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
    for i in [ref[1:] for ref in refs]:
        fig.update_layout({'yaxis' + i + '_title_text': ''})

    middle_ref_num = refs[len(refs) // 2][1:]
    fig.update_layout({'yaxis' + str(middle_ref_num) + '_title_text': 'subgroup'})
    fig.update_yaxes(showgrid=False, zeroline=False)

    return fig


def plot_fairness_check_clf(fobject, title=None, other_objects=None, epsilon=None, verbose=True):
    if other_objects is not None:
        basic_checks.check_other_fairness_objects(fobject, other_objects)
    objects = [fobject, *(other_objects or [])]
    frames = [long_table(obj).filter(nw.col('metric').is_in(['TPR', 'ACC', 'PPV', 'FPR', 'STP']) &
                                     (nw.col('subgroup') != obj.privileged)) for obj in objects]
    data = concat_frames(frames).with_columns((nw.col('score') - 1).alias('score'))
    if epsilon is not None:
        checks.check_epsilon(epsilon)
    zeros = data.filter(nw.col('score') == 0)
    if len(zeros):
        nan_models = set(zeros['label'].to_list())
        helper.verbose_cat(f"\nFound NaN's or 0's for models: {nan_models}\n"
                           "It is advisable to check 'metric_ratios'", verbose=verbose)
    data = rename_metrics(data, {
        'TPR': 'Equal opportunity ratio     TP/(TP + FN)',
        'ACC': 'Accuracy equality ratio    (TP + TN)/(TP + FP + TN + FN)',
        'PPV': 'Predictive parity ratio     TP/(TP + FP)',
        'FPR': 'Predictive equality ratio   FP/(FP + TN)',
        'STP': 'Statistical parity ratio   (TP + FP)/(TP + FP + TN + FN)'})
    return plot_fairness_check(data, len(objects), fobject.epsilon, title)


def plot_fairness_check_reg(fobject, title=None, other_objects=None, epsilon=None, verbose=True):
    if other_objects is not None:
        basic_checks.check_other_fairness_objects(fobject, other_objects)
    if epsilon is None:
        epsilon = fobject.epsilon
    else:
        checks.check_epsilon(epsilon)
    objects = [fobject, *(other_objects or [])]
    data = concat_frames([long_table(obj, row_major=False) for obj in objects])
    data = data.with_columns((nw.col('score') - 1).alias('score'))
    return plot_fairness_check(data, len(objects), epsilon, title)


def plot_metric_scores(fobject, other_objects, title=None, verbose=True):
    if other_objects is not None:
        basic_checks.check_other_fairness_objects(fobject, other_objects)
    objects = [fobject, *(other_objects or [])]
    data = concat_frames([long_table(obj, 'metric_scores').with_columns(nw.lit(str(obj.label)).alias('label'))
                          for obj in objects])
    data = data.filter(nw.col('metric').is_in(['TPR', 'PPV', 'STP', 'ACC', 'FPR']))
    data = rename_metrics(data, {
        'TPR': 'TPR    TP/(TP + FN)', 'ACC': 'ACC   (TP + TN)/(TP + FP + TN + FN)',
        'PPV': 'PPV    TP/(TP + FP)', 'FPR': 'FPR    FP/(FP + TN)', 'STP': 'STP   (TP + FP)/(TP + FP + TN + FN)'})
    min_score, max_score = finite_bounds(data['score'].to_numpy())
    n_ticks = len(unique(data, 'label'))
    tick_values = np.linspace(0, 1, n_ticks + 2)[1:-1]
    privileged_data = data.filter(nw.col('subgroup') == fobject.privileged)
    data = data.filter(nw.col('subgroup') != fobject.privileged)
    if not len(data):
        raise ParameterCheckError('No unprivileged subgroup scores are available to plot')
    subgroup_tick_dict = {sub: i for i, sub in enumerate(unique(data, 'subgroup'))}
    label_tick_dict = dict(zip(unique(data, 'label'), tick_values))
    coordinates = [subgroup_tick_dict[sub] + label_tick_dict[label]
                   for sub, label in zip(data['subgroup'].to_list(), data['label'].to_list())]
    data = data.with_columns(nw.new_series('subgroup_numeric', coordinates, backend=nw.get_native_namespace(data)))

    # drwhy colors
    colors = _theme.get_default_colors(len(unique(data, 'label')), 'line')

    # fig creation
    fig = px.scatter(data.to_native(),
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
    for label in unique(data, 'label'):
        color_dict[label] = colors[k]
        k += 1

    refs = ['y', 'y2', 'y3', 'y4', 'y5']
    refs_dict = {}
    j = 1
    for metric in unique(data, 'metric'):
        refs_dict[metric] = refs[len(refs) - j]
        j += 1
    # add lines
    lines_nan = False

    for metric in unique(data, 'metric'):
        for label in unique(data, 'label'):
            x = float(first(privileged_data.filter((nw.col('metric') == metric) &
                                                   (nw.col('label') == label)), 'score'))
            if not np.isfinite(x):
                lines_nan = True
                continue
            # lines
            for subgroup in unique(data, 'subgroup'):
                selected = data.filter((nw.col('metric') == metric) & (nw.col('label') == label) &
                                       (nw.col('subgroup') == subgroup))
                y = float(first(selected, 'subgroup_numeric'))
                x0 = float(first(selected, 'score'))
                if not np.isfinite(x0):
                    lines_nan = True
                    continue

                fig.add_shape(type='line',
                              xref='x',
                              yref=refs_dict.get(metric),
                              x0=x0,
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
                          y1=np.ceil(max(data['subgroup_numeric'].to_list())),
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
                     ticktext=list(subgroup_tick_dict_updated.keys()),
                     range=[-1/20 * len(subgroup_tick_dict), len(subgroup_tick_dict) + 1/20 * len(subgroup_tick_dict)])

    # delete y axis names [fixed] number of refs
    for i in ['', '2', '4', '5']:
        fig.update_layout({'yaxis' + i + '_title_text': ''})

    fig.update_layout({'yaxis3_title_text': 'subgroup'})

    # fixes rare bug where axis are in center and blank fields on left and right
    fig.update_xaxes(range=[min_score - 0.05, max_score + 0.05])
    fig.update_yaxes(showgrid=False, zeroline=False)

    # inform about nan's
    if lines_nan:
        data_joined = concat_frames([privileged_data.select('score', 'label'), data.select('score', 'label')])
        invalid = data_joined.filter(nw.col('score').is_null() | nw.col('score').is_nan())
        nan_labels = set(invalid['label'].to_list())

        helper.verbose_cat(f'\nFound NaN\'s while plotting metric_scores in models : {nan_labels}\n'
                           f'It is advisable to check \'metric_scores\' attribute of the GroupFairnessClassification',
                           verbose=verbose)

    return fig


def plot_stacked(fobject,
                 title=None,
                 other_objects=None,
                 metrics=["TPR", "PPV", "STP", "ACC", "FPR"],
                 verbose=True,
                 **kwargs):
    data = parity_data(fobject, other_objects, metrics, verbose)

    fig = px.bar(data.to_native(),
                 x='score',
                 y='label',
                 custom_data=[np.round(data['score'].to_numpy(), 3), data['metric'].to_list()],
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
    data = parity_data(fobject, other_objects, metrics, verbose)
    colors = _theme.get_default_colors(len(set(data['label'].to_list())), type='line')

    fig = go.Figure()
    for i, label in enumerate(set(sorted(data['label'].to_list()))):
        model_data = data.filter(nw.col('label') == label)
        r = model_data['score'].to_list()
        r.append(r[0])
        theta = model_data['metric'].to_list()
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
                range=[0, finite_bounds(data['score'].to_numpy(), default=(0., 0.))[1] + 0.1]
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
    data = parity_data(fobject, other_objects, [fairness_metric], verbose)
    assert len(unique(data, 'label')) == len(data)

    objects = [fobject, *(other_objects or [])]
    performance_data = nw.from_dict({'label': [obj.label for obj in objects],
        'performance_score': [utils._classification_performance(obj, verbose, performance_metric) for obj in objects]},
        backend=nw.get_native_namespace(data))
    data = (data.with_row_index('__row__').join(performance_data, on='label', how='inner')
            .sort('__row__').drop('__row__'))

    fig = px.scatter(data.to_native(),
                     x='performance_score',
                     y='score',
                     color='label',
                     custom_data=[data['label'].to_list()],
                     color_discrete_sequence=_theme.get_default_colors(len(data), 'line'))

    fig.update_traces(
        mode='markers',
        marker=dict(size=[15 for _ in range(len(data))]))

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
        metrics = table_frame(fobject, 'parity_loss')['metric'].to_list()

    data = parity_data(fobject, other_objects, metrics, verbose=verbose)
    score_data, model_labels, metric_labels = score_matrix(
        data, labels=[obj.label for obj in [fobject, *(other_objects or [])]])

    fig = px.imshow(score_data,
                    labels=dict(x="parity loss metrics", y="model", color="Metric's parity loss"),
                    x=metric_labels,
                    y=model_labels,
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

    if not isinstance(grid_points, (int, np.integer)) or grid_points < 2:
        raise ParameterCheckError('grid_points must be an integer of at least 2')

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
        cutoff = dict(object.cutoff)
        y = object.y
        y_hat = object.y_hat
        frames = []

        # generate data for hypothetical cutoffs
        for i in range(1, grid_points):
            cutoff[subgroup] = i / grid_points
            sub_confusion_matrix = utils.SubgroupConfusionMatrix(y_true=y,
                                                                 y_pred=y_hat,
                                                                 protected=protected,
                                                                 cutoff=cutoff)

            sub_confusion_matrix_metrics = utils.SubgroupConfusionMatrixMetrics(sub_confusion_matrix)
            parity_loss = utils.calculate_parity_loss(sub_confusion_matrix_metrics, privileged, native=True,
                                                      backend=getattr(object, '_result_backend', 'pandas'))
            frame = nw.from_native(parity_loss.data).filter(nw.col('metric').is_in(list(metrics)))
            frames.append(frame.with_columns(nw.lit(i / (grid_points - 1)).alias('cutoff')))

        data = concat_frames(frames)
        matrix, cutoffs, metric_names = score_matrix(data.with_columns(nw.col('cutoff').alias('label')))
        matrix = matrix[:, np.argsort(np.asarray(metric_names))]
        summed_metrics = matrix.sum(axis=1)
        finite = np.flatnonzero(np.isfinite(summed_metrics))
        if not len(finite):
            raise ParameterCheckError('No cutoff has finite parity loss for all requested metrics')
        min_index = finite[np.argmin(summed_metrics[finite])]
        min_cutoff = cutoffs[min_index]
        peak = finite_bounds(data['score'].to_numpy(), default=(0., 0.))[1]

        # make figure from individual parts
        for j, metric in enumerate(metrics):
            metric_data = data.filter(nw.col('metric') == metric)
            fig.add_trace(go.Scatter(x=metric_data['cutoff'].to_numpy(),
                                     y=metric_data['score'].to_numpy(),
                                     mode='lines',
                                     line=dict(color=colors[j]),
                                     name=metric,
                                     showlegend=k == 0),  # show only legend of first subplot
                          row=k + 1, col=1)

        fig.add_shape(type='line',
                      x0=min_cutoff,
                      x1=min_cutoff,
                      y0=0,
                      y1=peak,
                      layer='below',
                      line=dict(color="grey",
                                dash="dot"), row=k + 1, col=1)

        fig.add_annotation(
            x=min_cutoff,
            y=peak + 0.05,
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


def plot_density(fobject, other_objects, title):
    objects = [fobject, *(other_objects or [])]
    labels = list(dict.fromkeys(obj.label for obj in objects))
    subgroups = np.unique(np.concatenate([obj.protected for obj in objects])).tolist()
    if not subgroups:
        raise ParameterCheckError('No subgroup observations are available to plot')
    predictions = {(label, subgroup): np.concatenate([
        obj.y_hat[obj.protected == subgroup] for obj in objects if obj.label == label])
        for label in labels for subgroup in subgroups}

    fig = go.Figure()

    counter = 0
    for model in labels:
        for i, sub in enumerate(subgroups):
            counter += 1
            fig.add_trace(
                go.Violin(
                    box_visible=True,
                    x=predictions[model, sub],
                    y0=sub + model,
                    name=sub,
                    fillcolor=_theme.get_default_colors(len(subgroups), type='line')[i],
                    opacity=0.9,
                    line_color='black',
                    hoveron='points'

                )
            )

    violins_in_model = int(counter / len(labels))
    starter_violins = np.arange(0, counter, violins_in_model)

    fig.update_xaxes(title='prediction')
    fig.update_yaxes(title='model', tickvals=list((starter_violins + (violins_in_model - 1) / 2)),
                     ticktext=list(labels))

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
    fig.update_traces(hovertemplate="outlier prediction: %{x:.3f}")

    return fig
