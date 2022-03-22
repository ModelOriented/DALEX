from . import checks, plot, utils
from .._basics import checks as basic_checks
from .._basics._base_objects import _FairnessObject
from .._basics.exceptions import ParameterCheckError
from ... import _global_checks, _theme


class GroupFairnessClassification(_FairnessObject):

    def __init__(self, y, y_hat, protected, privileged, label, verbose=False, cutoff=0.5, epsilon=0.8):

        super().__init__(y, y_hat, protected, privileged, verbose)
        checks.check_classification_parameters(y, y_hat, protected, privileged, verbose)
        cutoff = checks.check_cutoff(self.protected, cutoff, verbose)
        self.cutoff = cutoff
        epsilon = checks.check_epsilon(epsilon)
        self.epsilon = epsilon

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

    def fairness_check(self, epsilon=None, verbose=True):
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
            (default is `0.8`, which is set during object initialization).
        verbose : bool
            Shows verbose text about potential problems 
            (e.g. `NaN` in model metrics that can cause misinterpretation).

        Returns
        -----------
        None (prints console output)

        """

        utils.universal_fairness_check(self,
                                       epsilon,
                                       verbose,
                                       num_for_not_fair=2,
                                       num_for_no_decision=1,
                                       metrics=utils.fairness_check_metrics())

    def plot(self,
             objects=None,
             type='fairness_check',
             title=None,
             show=True,
             **kwargs):
        """
        Parameters
        -----------
        objects : array_like of GroupFairnessClassification objects
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
            if not isinstance(objects, (list, tuple)):
                objects = [objects]
            for obj in objects:
                _global_checks.global_check_object_class(obj, self.__class__)
                other_objects.append(obj)
            basic_checks.check_other_fairness_objects(self, other_objects)

        if type == 'fairness_check':
            fig = plot.plot_fairness_check_clf(self,
                                               other_objects=other_objects,
                                               title=title, 
                                               **kwargs)

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
        elif type == 'density':
            fig = plot.plot_density(self,
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
            fig.show(config=_theme.get_default_config())
        else:
            return fig


class GroupFairnessRegression(_FairnessObject):

    def __init__(self, y, y_hat, protected, privileged, label, epsilon=0.8, verbose=False):

        super().__init__(y, y_hat, protected, privileged, verbose)
        checks.check_epsilon(epsilon)

        df_ratios = utils.calculate_regression_measures(y, y_hat, protected, privileged)

        self.result = df_ratios
        self.label = label
        self.epsilon = epsilon

    def fairness_check(self, epsilon=None, verbose=True):
        """Check if classifier passes various fairness criteria

        Fairness check is an easy way to check if the model is fair.
        For that, this method uses 3 non-discrimination criteria.
        The approximations are made to check the conditional independence expressed
        in form of independence, separation and sufficiency.
        Model is considered to be fair if all criteria are met.
        This arbitrary decision is based on epsilon,
        which defaults to `0.8` (it matches the four-fifths 80% rule).

        Methods in use: Independence, Separation, Sufficiency.

        Parameters
        -----------
        epsilon : float, optional
            Parameter defines acceptable fairness scores. The closer to `1` the
            more strict the verdict is. If the ratio of certain unprivileged
            and privileged subgroup is within the `(epsilon, 1/epsilon)` range,
            then there is no discrimination in this metric and for this subgroups
            (default is `0.8`, which is set during object initialization).
        verbose : bool
            Shows verbose text about potential problems
            (e.g. `NaN` in model metrics that can cause misinterpretation).

        Returns
        -----------
        None (prints console output)

        """

        utils.universal_fairness_check(self,
                                       epsilon,
                                       verbose,
                                       num_for_not_fair=1,
                                       num_for_no_decision=None,
                                       metrics=['independence', 'separation', 'sufficiency'])

    def plot(self, objects=None, type='fairness_check', title=None, show=True, **kwargs):
        """
        Parameters
        -----------
        objects : array_like of GroupFairnessRegression objects
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

            - density:
                density plot visualizes the output of the model for each
                subgroup in form of violin plots with boxplots on top of them.
                It does not accept additional keyword arguments.
        title : str, optional
            Title of the plot (default depends on the `type` attribute).

        """

        other_objects = None
        if objects is not None:
            other_objects = []
            if not isinstance(objects, (list, tuple)):
                objects = [objects]
            for obj in objects:
                _global_checks.global_check_object_class(obj, self.__class__)
                other_objects.append(obj)
            basic_checks.check_other_fairness_objects(self, other_objects)

        if type == 'density':
            fig = plot.plot_density(self,
                                    other_objects,
                                    title=title,
                                    **kwargs)

        elif type == 'fairness_check':
            fig = plot.plot_fairness_check_reg(self,
                                               other_objects=other_objects,
                                               title=title,
                                               **kwargs)

        else:
            raise ParameterCheckError(f"plot type {type} not supported, try other types.")

        if show:
            fig.show(config=_theme.get_default_config())
        else:
            return fig
