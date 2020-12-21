import numpy as np

from . import checks, plot, utils
from .._basics import checks as basic_checks
from .._basics._base_objects import _FairnessObject
from .._basics.exceptions import ParameterCheckError
from ..._theme import get_default_config
from ..._explainer import helper


class GroupFairnessClassification(_FairnessObject):

    def __init__(self, y, y_hat, protected, privileged, label, verbose=False, cutoff=0.5):

        super().__init__(y, y_hat, protected, privileged, verbose)

        cutoff = checks.check_cutoff(self.protected, cutoff, verbose)
        self.cutoff = cutoff

        sub_confusion_matrix = utils.SubgroupConfusionMatrix(
            y_true=self.y,
            y_pred=self.y_hat,
            protected=self.protected,
            cutoff=self.cutoff
        )

        sub_confusion_matrix_metrics = utils.SubgroupConfusionMatrixMetrics(sub_confusion_matrix)
        df_ratios = utils.calculate_ratio(sub_confusion_matrix_metrics, self.privileged)
        parity_loss = utils.calculate_parity_loss(sub_confusion_matrix_metrics, self.privileged)

        self._subgroup_confusion_matrix = sub_confusion_matrix
        self._subgroup_confusion_matrix_metrics_object = sub_confusion_matrix_metrics
        self.metric_scores = sub_confusion_matrix_metrics.to_horizontal_DataFrame()
        self.parity_loss = parity_loss
        self.result = df_ratios
        self.label = label

    def fairness_check(self, epsilon=0.8, verbose=True):
        """Check if classifier passes various fairness metrics

        Fairness check is an easy way to check if the model is fair.
        For that, this method uses 5 popular metrics of group fairness.
        Model is considered to be fair if confusion matrix metrics are
        close to each other. This arbitrary decision is based on epsilon,
        which defaults to `0.8` (it matches the four-fifths 80% rule).

        Methods in use: Equal opportunity, Predictive parity, Predictive equality,
        Statistical parity and Accuracy equality.

        Parameters
        -----------
        epsilon : float, optional
            Parameter defines acceptable fairness scores. The closer to `1` the
            more strict the verdict is. If the ratio of certain unprivileged
            and privileged subgroup is within the `(epsilon, 1/epsilon)` range,
            then there is no discrimination in this metric and for this subgroups
            (default is `0.8`).
        verbose : bool
            Shows verbose text about potential problems 
            (e.g. `NaN` in model metrics that can cause misinterpretation).

        Returns
        -----------
        None (prints console output)

        """
        epsilon = checks.check_epsilon(epsilon)
        metric_ratios = self.result

        subgroups = np.unique(self.protected)
        subgroups_without_privileged = subgroups[subgroups != self.privileged]
        metric_ratios = metric_ratios.loc[subgroups_without_privileged, utils.fairness_check_metrics()]

        metrics_exceeded = ((metric_ratios > 1 / epsilon) | (epsilon > metric_ratios)).apply(sum, 0)

        names_of_exceeded_metrics = list(metrics_exceeded.index[metrics_exceeded != 0])
        if len(names_of_exceeded_metrics) >= 2:
            print(f'Bias detected in {len(names_of_exceeded_metrics)} metrics: {", ".join(names_of_exceeded_metrics)}')
        elif len(names_of_exceeded_metrics) == 1:
            print(f'Bias detected in {len(names_of_exceeded_metrics)} metric: {names_of_exceeded_metrics[0]}')
        else:
            print("No bias was detected!")

        # arbitrary decision
        if len(names_of_exceeded_metrics) >= 2:
            conclusion = 'is not fair because 2 or more metric scores exceeded acceptable limits set by epsilon'
        elif len(names_of_exceeded_metrics) == 1:
            conclusion = 'cannot be called fair because 1 metric score exceeded acceptable limits set by epsilon.\n' \
                         'It does not mean that your model is unfair ' \
                         'but it cannot be automatically approved based on these metrics'
        else:
            conclusion = 'is fair in terms of checked fairness metrics'

        print(f'\nConclusion: your model {conclusion}.')

        print(
            f'\nRatios of metrics, based on \'{self.privileged}\'. Parameter \'epsilon\' was set to {epsilon}'
            f' and therefore metrics should be within ({epsilon}, {round(1 / epsilon, 3)})')
        print(utils.metric_ratios.to_string())
        if np.isnan(metric_ratios).sum().sum() > 0:
            helper.verbose_cat(
                '\nWarning!\nTake into consideration that NaN\'s are present, consider checking \'metric_scores\' '
                'plot to see the difference', verbose=verbose)

    def plot(self,
             objects=None,
             type='fairness_check',
             title=None,
             show=True,
             **kwargs):
        """
        Parameters
        -----------
        objects : GroupFairnessClassification object
            Additional objects to plot (default is `None`).
        type : str, optional
            Type of the plot. Default is `'fairness_check'`.
            When the type of plot is specified, user may provide additional
            keyword arguments (`**kwargs`) which will be used in creating
            plot of certain type. Below there is list of types:

            - fairness_check:
                fairness_check plot visualizes the fairness_check method
                for one or more GroupFairnessClassification objects.
                It accepts following keyword arguments:
                 'epsilon' - which denotes the decision
                             boundary (like in `fairness_check` method)
            - metric_scores:
                metric_scores plot shows real values of metrics.
                Each model displays values in each metric and each subgroup.
                Vertical lines show metric score for privileged
                subgroup and points connected with the lines
                show scores for unprivileged subgroups.
                This plot is simple and it does
                not have additional keyword arguments.
            - stacked:
                stacked plot shows cumulated parity loss from chosen
                metrics. It stacks metrics on top of each other.
                It accepts following keyword arguments:
                'metrics' - list of metrics to be plotted. The metrics are taken
                            from parity_loss attribute of the object.
                            Default is `["TPR", "ACC", "PPV", "FPR", "STP"]`.
            - radar:
                radar plot shows parity loss of provided metrics. It does it
                in form of radar (spider) chart. The smaller the field of
                figure the better.
                It accepts following keyword arguments:
                'metrics' - list of metrics to be plotted. The metrics are taken
                            from parity_loss attribute of the object.
                            Default is `["TPR", "ACC", "PPV", "FPR", "STP"]`.
            - performance_and_fairness:
                performance_and_fairness plot shows relation between chosen
                performance and fairness metrics. The fairness metric axis is
                reversed, because the higher the model the less bias it has.
                Thanks to that it is more intuitive to look at because
                the best models are in top right corner.
                It accepts following keyword arguments:
                'fairness_metric' - single fairness metric to be plotted on Y axis.
                                   The metric is taken from parity_loss attribute\
                                   of the object. The default is "TPR"
                'performance_metric' - single performance metric. One of `{'recall',
                                       'precision','accuracy','auc','f1'}`.
                                       Metrics apart from 'auc' are
                                       cutoff-sensitive. Default is "accuracy"
            - heatmap:
                heatmap shows parity loss of metrics in form of heatmap. The less
                parity loss model has, the more fair it is.
                It accepts following keyword arguments:
                'metrics' - list of metrics to be plotted. The metrics are taken
                            from parity_loss attribute of the object.
                            Default is 'all' which stands for all available metrics.
            - ceteris_paribus_cutoff:
                ceteris_paribus_cutoff plot shows what would happen if cutoff
                for only one subgroup would change with others cutoffs constant.
                The plot shows also a minimum, where sum of parity loss of metrics
                is the lowest. Minimum only works if at some interval all metrics
                have non-nan scores.
                It accepts following keyword arguments:
                'subgroup' - necessary argument. It is name of subgroup from
                             protected attribute. Cutoff for this subgroup will
                             be changed.

                'metrics' - list of metrics to be plotted. The metrics are taken
                            from parity_loss attribute of the object.
                            Default is `["TPR", "ACC", "PPV", "FPR", "STP"]`.

                'grid_points' - number of grid points (cutoff values) to calculate
                                metrics for. The points are distributed evenly.
                                Default is `101`.

        title : str, optional
            Title of the plot (default depends on the `type` attribute).
        show : bool, optional
            `True` shows the plot; `False` returns the plotly Figure object that can
            be edited or saved using the `write_image()` method (default is `True`).

        Returns
        -----------
        None or plotly.graph_objects.Figure
            Return figure that can be edited or saved. See `show` parameter.
        """
        other_objects = None
        if objects is not None:
            other_objects = []
            for obj in objects:
                if isinstance(obj, self.__class__):
                    other_objects.append(obj)
            basic_checks.check_other_fairness_objects(self, other_objects)

        if type == 'fairness_check':
            fig = plot.plot_fairness_check(self,
                                      other_objects=other_objects,
                                      title=title, **kwargs)

        elif type == "metric_scores":
            fig = plot.plot_metric_scores(self,
                                     other_objects=other_objects,
                                     title=title,
                                     **kwargs)

        # names of plots may be changed
        elif type == 'stacked':
            fig = plot.plot_stacked(self,
                               other_objects=other_objects,
                               title=title,
                               **kwargs)

        elif type == 'radar':
            fig = plot.plot_radar(self,
                             other_objects=other_objects,
                             title=title,
                             **kwargs)

        elif type == 'performance_and_fairness':
            fig = plot.plot_performance_and_fairness(self,
                                                other_objects=other_objects,
                                                title=title,
                                                **kwargs)

        elif type == 'heatmap':
            fig = plot.plot_heatmap(self,
                               other_objects=other_objects,
                               title=title,
                               **kwargs)

        elif type == 'ceteris_paribus_cutoff':
            fig = plot.plot_ceteris_paribus_cutoff(self,
                                              other_objects=other_objects,
                                              title=title,
                                              **kwargs)

        else:
            raise ParameterCheckError(f"plot type {type} not supported, try other types.")

        if show:
            fig.show(config=get_default_config())
        else:
            return fig
