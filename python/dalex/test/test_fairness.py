import unittest

from copy import deepcopy
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

import dalex as dx
from plotly.graph_objs import Figure


class FairnessTest(unittest.TestCase):

    def setUp(self):
        self.protected = np.array(['a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'c'])
        self.y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        self.y_pred = np.array([0.32, 0.43, 0.56, 0.67, 0.9, 0.67, 0.98, 0.1, 0.44, 1, 0.65, 0.55, 1])
        self.cutoff = {'a': 0.5, 'b': 0.4, 'c': 0.6}

        data = dx.datasets.load_german()

        X = data.drop(columns='risk')
        y = data.risk

        categorical_features = ['sex', 'job', 'housing', 'saving_accounts', "checking_account", 'purpose']
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features)])

        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', DecisionTreeClassifier(max_depth=7, random_state=123))])
        clf.fit(X, y)

        self.exp = dx.Explainer(clf, X, y, verbose=False)
        self.german_protected = data.sex + '_' + np.where(data.age < 25, 'young', 'old')

    def test_ConfusionMatrix(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0.32, 0.54, 0.56, 0.67, 0.34, 0.67, 0.98, 1])
        cutoff = 0.55
        cm = dx.fairness._group_fairness.utils.ConfusionMatrix(y_true, y_pred, cutoff)

        #  proper calculations
        self.assertEqual(cm.cutoff, 0.55)
        self.assertEqual(cm.tp, 3)
        self.assertEqual(cm.tn, 2)
        self.assertEqual(cm.fp, 2)
        self.assertEqual(cm.fn, 1)

        #  error assertions
        y_true = y_true[:-1]
        with self.assertRaises(AssertionError):
            cm_ = dx.fairness._group_fairness.utils.ConfusionMatrix(y_true, y_pred, cutoff)
        y_true = np.append(y_true, 1)
        cutoff = 1.5
        with self.assertRaises(AssertionError):
            cm_ = dx.fairness._group_fairness.utils.ConfusionMatrix(y_true, y_pred, cutoff)

    def test_SubConfusionMatrix(self):

        y_true = self.y_true.copy()
        y_pred = self.y_pred
        protected = self.protected
        cutoff = self.cutoff

        scf = dx.fairness._group_fairness.utils.SubgroupConfusionMatrix(y_true, y_pred, protected, cutoff)

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
            cm_ = dx.fairness._group_fairness.utils.SubgroupConfusionMatrix(y_true, y_pred, protected, cutoff)
        y_true = np.append(y_true, 1)
        cutoff = [0.1, 0.2, 0.4]  # list instead of dict
        with self.assertRaises(AssertionError):
            cm_ = dx.fairness._group_fairness.utils.SubgroupConfusionMatrix(y_true, y_pred, protected, cutoff)

    def test_SubgroupConfusionMatrixMetrics(self):
        y_true = self.y_true
        y_pred = self.y_pred
        protected = self.protected
        cutoff = self.cutoff

        scf = dx.fairness._group_fairness.utils.SubgroupConfusionMatrix(y_true, y_pred, protected, cutoff)
        scf_metrics = dx.fairness._group_fairness.utils.SubgroupConfusionMatrixMetrics(scf)
        scmm = scf_metrics.subgroup_confusion_matrix_metrics
        self.assertEqual(scmm.get('a').get('TPR'), 1)
        self.assertEqual(scmm.get('b').get('TPR'), 0.667)
        self.assertTrue(np.isnan(scmm.get('b').get('TNR')))

    def test_calculate_ratio(self):
        y_true = self.y_true
        y_pred = self.y_pred
        protected = self.protected
        cutoff = self.cutoff

        scf = dx.fairness._group_fairness.utils.SubgroupConfusionMatrix(y_true, y_pred, protected, cutoff)
        scf_metrics = dx.fairness._group_fairness.utils.SubgroupConfusionMatrixMetrics(scf)

        df_ratio = dx.fairness._group_fairness.utils.calculate_ratio(scf_metrics, 'a')

        b = list(scf_metrics.subgroup_confusion_matrix_metrics.get('b').values())
        a = list(scf_metrics.subgroup_confusion_matrix_metrics.get('a').values())

        ratio = np.array(b) / np.array(a)
        ratio[np.isinf(ratio)] = np.nan
        ratio[ratio == 0] = np.nan

        ratio_nonnan = ratio[np.isfinite(ratio)]
        df_ratio_nonnan = np.array(df_ratio.iloc[1, :][np.isfinite(df_ratio.iloc[1, :])])

        self.assertTrue(np.equal(ratio_nonnan, df_ratio_nonnan).all())

    def test_calculate_parity_loss(self):
        y_true = self.y_true
        y_pred = self.y_pred
        protected = self.protected
        cutoff = self.cutoff

        scf = dx.fairness._group_fairness.utils.SubgroupConfusionMatrix(y_true, y_pred, protected, cutoff)
        scf_metrics = dx.fairness._group_fairness.utils.SubgroupConfusionMatrixMetrics(scf)

        parity_loss = dx.fairness._group_fairness.utils.calculate_parity_loss(scf_metrics, "a")
        ratios = dx.fairness._group_fairness.utils.calculate_ratio(scf_metrics, "a")
        TPR_parity_loss = parity_loss.iloc[0]

        TPR_ratios = ratios.TPR / ratios.TPR[0]
        TPR_log = abs(np.log(TPR_ratios))

        self.assertEqual(TPR_log.sum(), TPR_parity_loss)

    def test_GroupFairnessClassification(self):
        exp = self.exp
        protected = self.german_protected

        gfco = dx.fairness.GroupFairnessClassification(y=exp.y,
                                                       y_hat=exp.y_hat,
                                                       protected=protected,
                                                       privileged='male_old',
                                                       verbose=False,
                                                       label=exp.label)
        self.assertIsInstance(gfco, dx.fairness.GroupFairnessClassification)

    def test_parameter_checks(self):
        exp = self.exp
        protected = self.german_protected

        #  error handling
        wrong_protected = np.array([protected, protected])
        with self.assertRaises(dx.fairness._basics.exceptions.ParameterCheckError):
            gfco = exp.model_fairness(protected=wrong_protected,
                                      privileged='male_old',
                                      verbose=False)

        with self.assertRaises(dx.fairness._basics.exceptions.ParameterCheckError):
            gfco = exp.model_fairness(protected=protected,
                                      privileged='not_existing',
                                      verbose=False)
        with self.assertRaises(dx.fairness._basics.exceptions.ParameterCheckError):
            gfco = dx.fairness.GroupFairnessClassification(y=exp.y[:-1, ],
                                                           y_hat=exp.y_hat,
                                                           protected=protected,
                                                           privileged='male_old',
                                                           verbose=False,
                                                           label=exp.label)
        with self.assertRaises(dx.fairness._basics.exceptions.ParameterCheckError):
            gfco = dx.fairness.GroupFairnessClassification(y=exp.y[:-1, ],
                                                           y_hat=exp.y_hat[:-1, ],
                                                           protected=protected,
                                                           privileged='male_old',
                                                           verbose=False,
                                                           label=exp.label)

        with self.assertRaises(dx.fairness._basics.exceptions.ParameterCheckError):
            gfco = exp.model_fairness(protected=protected,
                                      privileged='male_old',
                                      cutoff=1.2,
                                      verbose=False)

        with self.assertRaises(dx.fairness._basics.exceptions.ParameterCheckError):
            gfco = exp.model_fairness(protected=protected,
                                      privileged='male_old',
                                      cutoff='not_int',
                                      verbose=False)

        # conversion check
        gfco = exp.model_fairness(protected=protected,
                                  privileged='male_old',
                                  cutoff=0.6,
                                  verbose=False)

        self.assertEqual(list(gfco.cutoff.values()), [0.6, 0.6, 0.6, 0.6])

        gfco = exp.model_fairness(protected=protected,
                                  privileged='male_old',
                                  cutoff={'male_old': 0.9},
                                  verbose=False)
        self.assertEqual(gfco.cutoff, {'male_old': 0.9, 'female_old': 0.5, 'male_young': 0.5, 'female_young': 0.5})

        np.random.seed(1)
        new_protected = np.random.choice(np.array([0, 1]), 1000)
        gfco = exp.model_fairness(protected=new_protected,
                                  privileged=1,
                                  verbose=False)

        self.assertEqual(gfco.privileged, '1')
        self.assertEqual(list(gfco.protected), list(new_protected.astype('U')))

        gfco = exp.model_fairness(protected=list(protected),
                                  privileged='male_old',
                                  verbose=False)

        self.assertEqual(type(gfco.protected), np.ndarray)
        self.assertIsInstance(gfco, dx.fairness.GroupFairnessClassification)

    def test_model_group_fairness(self):
        exp = self.exp
        protected = self.german_protected
        mgf = exp.model_fairness(protected=protected,
                                 privileged='male_old',
                                 verbose=False)

        self.assertIsInstance(mgf, dx.fairness.GroupFairnessClassification)

    def test_plot_fairness_check(self):
        exp = self.exp
        protected = self.german_protected
        mgf = exp.model_fairness(protected=protected,
                                 privileged='male_old',
                                 verbose=False)
        fig1 = mgf.plot(show=False)
        self.assertEqual(fig1.layout.title.text, "Fairness Check")
        self.assertIsInstance(fig1, Figure)

        mgf2 = deepcopy(mgf)
        mgf.label = 'first'
        mgf2.label = 'second'

        fig2 = mgf.plot(objects=[mgf2], show=False)
        self.assertIsInstance(fig2, Figure)

        self.assertEqual(fig2['data'][0]['legendgroup'], "first")
        self.assertEqual(fig2['data'][5]['legendgroup'], "second")

        # test errors in plots
        with self.assertRaises(dx.fairness._basics.exceptions.FairnessObjectsDifferenceError):
            mgf_wrong = exp.model_fairness(protected=protected,
                                           privileged='male_young',
                                           verbose=False
                                           )
            mgf.plot([mgf_wrong])

        with self.assertRaises(dx.fairness._basics.exceptions.FairnessObjectsDifferenceError):
            exp_wrong = deepcopy(exp)
            exp_wrong.y = exp_wrong.y[:-1]
            exp_wrong.y_hat = exp_wrong.y_hat[:-1]

            mgf_wrong = exp_wrong.model_fairness(protected=protected[:-1],
                                                 privileged='male_old',
                                                 verbose=False)
            mgf.plot([mgf_wrong])

    def test_plot_metric_scores(self):
        exp = self.exp
        protected = self.german_protected
        mgf = exp.model_fairness(protected=protected,
                                 privileged='male_old',
                                 verbose=False)
        fig1 = mgf.plot(show=False, type='metric_scores')
        self.assertEqual(fig1.layout.title.text, "Metric Scores")
        self.assertIsInstance(fig1, Figure)

        mgf2 = deepcopy(mgf)
        mgf.label = 'first'
        mgf2.label = 'second'

        fig2 = mgf.plot(objects=[mgf2], show=False, type='metric_scores')
        self.assertIsInstance(fig2, Figure)

        self.assertEqual(fig2['data'][0]['legendgroup'], "first")
        self.assertEqual(fig2['data'][5]['legendgroup'], "second")

        # test errors in plots
        with self.assertRaises(dx.fairness._basics.exceptions.FairnessObjectsDifferenceError):
            mgf_wrong = exp.model_fairness(protected=protected,
                                           privileged='male_young',
                                           verbose=False)
            mgf.plot([mgf_wrong], type='metric_scores')

        with self.assertRaises(dx.fairness._basics.exceptions.FairnessObjectsDifferenceError):
            exp_wrong = deepcopy(exp)
            exp_wrong.y = exp_wrong.y[:-1]
            exp_wrong.y_hat = exp_wrong.y_hat[:-1]

            mgf_wrong = exp_wrong.model_fairness(protected=protected[:-1],
                                                 privileged='male_old',
                                                 verbose=False)
            mgf.plot([mgf_wrong], type='metric_scores')


if __name__ == '__main__':
    unittest.main()
