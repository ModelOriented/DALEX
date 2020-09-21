from .checks import *
from .utils import *
from ..basics._base_objects import _FairnessObject


class GroupFairnessClassificationObject(_FairnessObject):

    def __init__(self, y, y_hat, protected, privileged, verbose=True, cutoff=0.5):
        super().__init__(y, y_hat, protected, privileged, verbose)

        cutoff = check_cutoff(protected, cutoff, verbose)
        self.cutoff = cutoff

        sub_confusion_matrix = SubgroupConfusionMatrix(y_true=self.y,
                                                       y_pred=self.y_hat,
                                                       protected=self.protected,
                                                       cutoff=self.cutoff)

        sub_confusion_matrix_metrics = SubgroupConfusionMatrixMetrics(sub_confusion_matrix)
        df_ratios = calculate_ratio(sub_confusion_matrix_metrics, privileged)
        parity_loss = calculate_parity_loss(sub_confusion_matrix_metrics, privileged)

        self.subgroup_metrics = sub_confusion_matrix_metrics
        self.parity_loss = parity_loss
        self.metric_ratios = df_ratios

    def fairness_check(self, epsilon=0.8):
        """Check if classifier passes popular fairness metrics

        Fairness check is easy way to check if model is fair. For that method uses 5 popular
        metrics of group fairness. Model is considered to be fair if confusion matrix metrics are
        close to each other. This arbitrary decision is based on epsilon, which default value
        is 0.8 which matches four-fifths (80%) rule.

        Methods in use: Equal opportunity, Predictive parity, Predictive equality,
        Statistical parity and Accuracy equality.

        Parameters
        -----------
        epsilon : float (default 0.8)
            Parameter defines acceptable fairness scores. The closer to 1 the more strict the vardict is.
            If the ratio of certain unprivileged and privileged subgroup is within (epsilon, 1/epsilon) than
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
        for rowname in metrics_exceeded.index :
            print(f'{rowname}, metrics exceeded: {metrics_exceeded[rowname]}')

        print(f'\nRatio values: \n')
        print(metric_ratios.to_string())

        if sum(metrics_exceeded) > 2 :
            conclusion = 'not fair'
        else :
            conclusion = 'fair'

        print(f'\nConclusion: your model is {conclusion}')
        return

