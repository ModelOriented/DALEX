from .checks import *
from .plot import *
from ..basics._base_objects import _FairnessObject
from ..basics.checks import check_other_objects


class GroupFairnessClassification(_FairnessObject):

    def __init__(self, y, y_hat, protected, privileged, label, verbose=True, cutoff=0.5):

        super().__init__(y, y_hat, protected, privileged, verbose)

        cutoff = check_cutoff(self.protected, cutoff, verbose)
        self.cutoff = cutoff

        sub_confusion_matrix = SubgroupConfusionMatrix(y_true=self.y,
                                                       y_pred=self.y_hat,
                                                       protected=self.protected,
                                                       cutoff=self.cutoff)

        sub_confusion_matrix_metrics = SubgroupConfusionMatrixMetrics(sub_confusion_matrix)
        df_ratios = calculate_ratio(sub_confusion_matrix_metrics, self.privileged)
        parity_loss = calculate_parity_loss(sub_confusion_matrix_metrics, self.privileged)

        self._subgroup_confusion_matrix_metrics_object = sub_confusion_matrix_metrics
        self.subgroup_metrics = sub_confusion_matrix_metrics.to_horizontal_DataFrame()
        self.parity_loss = parity_loss
        self.metric_ratios = df_ratios
        self.label = label

    def fairness_check(self, epsilon=0.8, verbose=True):
        """Check if classifier passes popular fairness metrics

        Fairness check is easy way to check if model is fair.
        For that method uses 5 popular metrics of group fairness.
        Model is considered to be fair if confusion matrix
        metrics are close to each other.
        This arbitrary decision is based on epsilon, which default value
        is 0.8 which matches four-fifths (80%) rule.

        Methods in use: Equal opportunity, Predictive parity, Predictive equality,
        Statistical parity and Accuracy equality.

        Parameters
        -----------
        epsilon : float (default 0.8)
            Parameter defines acceptable fairness scores. The closer to 1 the
            more strict the vardict is. If the ratio of certain unprivileged
            and privileged subgroup is within (epsilon, 1/epsilon) than
            there is no discrimination in this metric and for this subgroups.

        Returns
        -----------
        Console output

        """
        epsilon = check_epsilon(epsilon)
        metric_ratios = self.metric_ratios

        subgroups = np.unique(self.protected)
        subgroups_without_privileged = subgroups[subgroups != self.privileged]
        metric_ratios = metric_ratios.loc[subgroups_without_privileged, fairness_check_metrics()]

        metrics_exceeded = ((metric_ratios > 1 / epsilon) | (epsilon > metric_ratios)).apply(sum, 1)

        print(f'\nRatios of metrics, base: {self.privileged}')
        for rowname in metrics_exceeded.index:
            print(f'{rowname}, metrics exceeded: {metrics_exceeded[rowname]}')

        print(f'\nRatio values: \n')
        print(metric_ratios.to_string())

        if sum(metrics_exceeded) >= 2:
            conclusion = 'not fair'
        else:
            conclusion = 'fair'

        print(f'\nConclusion: your model is {conclusion}')

        if np.isnan(metric_ratios).sum().sum() > 0:
            verbose_cat(
                '\nWarning!\nTake into consideration that NaN\'s are present, consider checking \'metric_scores\' '
                'plot to see the difference', verbose=verbose)

        return

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
            Additional objects to plot (default is None).
        type : str, optional
            Type of the plot. Default is 'fairness_check'.
            When the type of plot is specified, user may provide additional
            keyword arguments (**kwargs) which will
            be used in creating plot of certain type.
            Below there is list of types and **kwargs used by them

            fairness_check:
                fairness_check plot visualizes the fairness_check method
                for one or more GroupFairnessClassification objects.
                It accepts following keyword arguments:
                 'epsilon' - which denotes the decision
                             boundary (like in fairness_check method)
            metric_scores:
                metric_scores plot shows real values of metrics.
                Each model displays values in each metric and each subgroup.
                Vertical lines show metric score for privileged
                subgroup and points connected with the lines
                show scores for unprivileged subgroups.
                This plot is simple and it does
                not have additional keyword arguments.

        title : str, optional
            Title of the plot (default depends on the `type` attribute).
        show : bool, optional
            True shows the plot; False returns the plotly Figure object that can be
            edited or saved using the `write_image()` method (default is True).

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
        if other_objects is not None:
            check_other_objects(self, other_objects)

        config_fairness_plot = {'displaylogo': False, 'staticPlot': False,
                                'toImageButtonOptions': {'height': None, 'width': None, },
                                'modeBarButtonsToRemove': ['sendDataToCloud', 'lasso2d', 'autoScale2d', 'select2d',
                                                           'zoom2d',
                                                           'pan2d', 'zoomIn2d', 'zoomOut2d', 'resetScale2d',
                                                           'toggleSpikelines', 'hoverCompareCartesian',
                                                           'hoverClosestCartesian']}

        if type == 'fairness_check':
            fig = plot_fairness_check(self,
                                      other_objects=other_objects,
                                      title=title, **kwargs)

            if show:
                fig.show(config=config_fairness_plot)
            else:
                return fig

        if type == "metric_scores":
            fig = plot_metric_scores(self,
                                     other_objects=other_objects,
                                     title=title,
                                     **kwargs)
            if show:
                fig.show(config=config_fairness_plot)
            else:
                return fig
