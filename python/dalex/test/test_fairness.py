import unittest
import numpy as np
from dalex.fairness.group_fairness.utils import *



class FairnessTest(unittest.TestCase):

    def test_ConfusionMatrix(self):
        from dalex.fairness.group_fairness.utils import ConfusionMatrix
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0.32, 0.54, 0.56, 0.67, 0.34, 0.67, 0.98, 1])
        cutoff = 0.55
        cm = ConfusionMatrix(y_true, y_pred, cutoff)

        #  proper calculations
        self.assertEqual(cm.cutoff, 0.55)
        self.assertEqual(cm.tp, 3)
        self.assertEqual(cm.tn, 2)
        self.assertEqual(cm.fp, 2)
        self.assertEqual(cm.fn, 1)

        #  error assertions
        y_true = y_true[:-1]
        with self.assertRaises(AssertionError):
            cm_ = ConfusionMatrix(y_true, y_pred, cutoff)
        y_true = np.append(y_true, 1)
        cutoff = 1.5
        with self.assertRaises(AssertionError):
            cm_ = ConfusionMatrix(y_true, y_pred, cutoff)

    protected = np.array(['a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'c'])
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    y_pred = np.array([0.32, 0.43, 0.56, 0.67, 0.9, 0.67, 0.98, 0.1, 0.44, 1, 0.65, 0.55, 1])
    cutoff = {'a': 0.5, 'b': 0.4, 'c': 0.6}


    def test_SubConfusionMatrix(self):
        from dalex.fairness.group_fairness.utils import SubgroupConfusionMatrix

        y_true = FairnessTest.y_true
        y_pred = FairnessTest.y_pred
        protected = FairnessTest.protected
        cutoff = FairnessTest.cutoff

        scf = SubgroupConfusionMatrix(y_true, y_pred, protected, cutoff)

        #  proper calculations
        self.assertEqual(scf.sub_dict.get('a').tp, 2)
        self.assertEqual(scf.sub_dict.get('a').fp, 2)
        self.assertEqual(scf.sub_dict.get('a').fn, 0)
        self.assertEqual(scf.sub_dict.get('a').tn, 2)

        self.assertEqual(scf.sub_dict.get('b').tp, 2)
        self.assertEqual(scf.sub_dict.get('b').fp, 0)
        self.assertEqual(scf.sub_dict.get('b').fn, 1)
        self.assertEqual(scf.sub_dict.get('b').tn, 0)

        self.assertEqual(scf.sub_dict.get('c').tp, 2)
        self.assertEqual(scf.sub_dict.get('c').fp, 1)
        self.assertEqual(scf.sub_dict.get('c').fn, 0)
        self.assertEqual(scf.sub_dict.get('c').tn, 1)

        #  error assertions
        y_true = y_true[:-1]
        with self.assertRaises(AssertionError):
            cm_ = SubgroupConfusionMatrix(y_true, y_pred, protected, cutoff)
        y_true = np.append(y_true, 1)
        cutoff = [0.1, 0.2, 0.4]  # list instead of dict
        with self.assertRaises(AssertionError):
            cm_ = SubgroupConfusionMatrix(y_true, y_pred, protected, cutoff)

    def test_SubgroupConfusionMatrixMetrics(self):

        y_true = FairnessTest.y_true
        y_pred = FairnessTest.y_pred
        protected = FairnessTest.protected
        cutoff = FairnessTest.cutoff

        scf = SubgroupConfusionMatrix(y_true, y_pred, protected, cutoff)
        scf_metrics = SubgroupConfusionMatrixMetrics(scf)
        scmm = scf_metrics.subgroup_confusion_matrix_metrics
        self.assertEqual(scmm.get('a').get('TPR'), 1)
        self.assertEqual(scmm.get('b').get('TPR'), 0.667)
        self.assertTrue(np.isnan(scmm.get('b').get('TNR')))

    def test_calculate_ratio(self):


        y_true = FairnessTest.y_true
        y_pred = FairnessTest.y_pred
        protected = FairnessTest.protected
        cutoff = FairnessTest.cutoff

        scf = SubgroupConfusionMatrix(y_true, y_pred, protected, cutoff)
        scf_metrics = SubgroupConfusionMatrixMetrics(scf)

        df_ratio = calculate_ratio(scf_metrics, 'a')

        b = list(scf_metrics.subgroup_confusion_matrix_metrics.get('b').values())
        a = list(scf_metrics.subgroup_confusion_matrix_metrics.get('a').values())

        ratio = np.array(b)/np.array(a)
        ratio[np.isinf(ratio)] = np.nan
        ratio[ratio == 0] = np.nan

        ratio_nonnan = ratio[np.isfinite(ratio)]
        df_ratio_nonnan = np.array(df_ratio.iloc[1,:][np.isfinite(df_ratio.iloc[1,:])])

        self.assertTrue(np.equal(ratio_nonnan, df_ratio_nonnan).all())

    def test_calculate_parity_loss(self):

        y_true = FairnessTest.y_true
        y_pred = FairnessTest.y_pred
        protected = FairnessTest.protected
        cutoff = FairnessTest.cutoff

        scf = SubgroupConfusionMatrix(y_true, y_pred, protected, cutoff)
        scf_metrics = SubgroupConfusionMatrixMetrics(scf)

        parity_loss = calculate_parity_loss(scf_metrics, "a")
        ratios = calculate_ratio(scf_metrics, "a")
        TPR_parity_loss = parity_loss.iloc[0]

        TPR_ratios = ratios.TPR/ratios.TPR[0]
        TPR_log = np.log(TPR_ratios)

        self.assertEqual(TPR_log.sum(), TPR_parity_loss)


    def test_GroupFairnessObject(self):
        # TODO
        pass

    def test_model_group_fairness(self):
        # TODO
        pass

if __name__ == '__main__':
    unittest.main()
