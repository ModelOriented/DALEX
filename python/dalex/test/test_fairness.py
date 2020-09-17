import unittest
import numpy as np




class FairnessTest(unittest.TestCase):

    def test_ConfusionMatrix(self):
        from dalex.fairness.group_fairness.utils import _ConfusionMatrix
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0.32, 0.54, 0.56, 0.67, 0.34, 0.67, 0.98, 1])
        cutoff = 0.55
        cm = _ConfusionMatrix(y_true, y_pred, cutoff)

        #  proper calculations
        self.assertEqual(cm.cutoff, 0.55)
        self.assertEqual(cm.tp, 3)
        self.assertEqual(cm.tn, 2)
        self.assertEqual(cm.fp, 2)
        self.assertEqual(cm.fn, 1)

        #  error assertions
        y_true = y_true[:-1]
        with self.assertRaises(AssertionError):
            cm_ = _ConfusionMatrix(y_true, y_pred, cutoff)
        y_true = np.append(y_true, 1)
        cutoff = 1.5
        with self.assertRaises(AssertionError):
            cm_ = _ConfusionMatrix(y_true, y_pred, cutoff)

    protected = np.array(['a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'c'])
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    y_pred = np.array([0.32, 0.43, 0.56, 0.67, 0.9, 0.67, 0.98, 0.1, 0.44, 1, 0.65, 0.55, 1])
    cutoff = {'a': 0.5, 'b': 0.4, 'c': 0.6}


    def test_SubConfusionMatrix(self):
        from dalex.fairness.group_fairness.utils import _SubConfusionMatrix

        y_true = FairnessTest.y_true
        y_pred = FairnessTest.y_pred
        protected = FairnessTest.protected
        cutoff = FairnessTest.cutoff

        scf = _SubConfusionMatrix(y_true, y_pred, protected, cutoff)

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
            cm_ = _SubConfusionMatrix(y_true, y_pred, protected,  cutoff)
        y_true = np.append(y_true, 1)
        cutoff = [0.1, 0.2, 0.4]  # list instead of dict
        with self.assertRaises(AssertionError):
            cm_ = _SubConfusionMatrix(y_true, y_pred, protected,  cutoff)

    def test_SubgroupConfusionMatrixMetrics(self):
        from dalex.fairness.group_fairness.utils import _SubConfusionMatrix
        from dalex.fairness.group_fairness.utils import _SubroupConfusionMatrixMetrics

        y_true = FairnessTest.y_true
        y_pred = FairnessTest.y_pred
        protected = FairnessTest.protected
        cutoff = FairnessTest.cutoff

        scf = _SubConfusionMatrix(y_true, y_pred, protected, cutoff)
        scf_metrics = _SubroupConfusionMatrixMetrics(scf)

        # @TODO self.assertEqual()

    def test_metrics(self):
        # @TODO
        pass


if __name__ == '__main__':
    unittest.main()
