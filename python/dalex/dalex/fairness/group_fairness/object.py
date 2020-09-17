from .utils import *
from .checks import *

class GroupFairnessObject:

    def __init__(self, y, y_hat, protected, privileged, cutoff = 0.5):

        y, y_hat, protected, privileged, cutoff = check_parameters(y, y_hat, protected, privileged, cutoff)

        self.cutoff = cutoff
        self.privileged = privileged
        self.protected = protected
        self.y_hat = y_hat
        self.y = y

        sub_confusion_matrix = SubgroupConfusionMatrix(y_true=y, y_pred=y_hat, protected=protected, cutoff=cutoff)
        sub_confusion_matrix_metrics = SubgroupConfusionMatrixMetrics(sub_confusion_matrix)
        df_ratios = calculate_ratio(sub_confusion_matrix_metrics, privileged)
        parity_loss = calculate_parity_loss(sub_confusion_matrix_metrics, privileged)

        self.parity_loss = parity_loss
        self.metric_ratios = df_ratios


    def fairness_check(self, epsilon = 0.8):
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

        epsilon =  check_epsilon(epsilon)
        metric_ratios = self.metric_ratios

        subgroups = np.unique(self.protected)
        subgroups_without_privileged = subgroups[subgroups != self.privileged]
        metric_ratios = metric_ratios.loc[subgroups_without_privileged, fairness_check_metrics()]
        # TODO : finish the console output here

