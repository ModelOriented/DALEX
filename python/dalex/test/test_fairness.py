import unittest
from copy import deepcopy

import numpy as np
import pandas as pd
from copy import copy
from plotly.graph_objs import Figure
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

import dalex as dx
from dalex.fairness import reweight, resample, roc_pivot


class FairnessTest(unittest.TestCase):

    def setUp(self):
        self.protected = np.array(['a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'c'])
        self.y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        self.y_pred = np.array([0.32, 0.43, 0.56, 0.67, 0.9, 0.67, 0.98, 0.1, 0.44, 1, 0.65, 0.55, 1])
        self.cutoff = {'a': 0.5, 'b': 0.4, 'c': 0.6}

        # classifier
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

        self.mgf = self.exp.model_fairness(protected=self.german_protected,
                                           privileged='male_old',
                                           verbose=False)
        self.mgf2 = deepcopy(self.mgf)
        self.mgf.label = 'first'
        self.mgf2.label = 'second'

        # regressor
        self.protected_reg = np.array([np.tile('A', 1000), np.tile('B', 1000)]).flatten()
        first = np.array([np.random.normal(100, 20, 1000), np.random.normal(50, 10, 1000)]).flatten()
        second = np.array([np.random.normal(60, 20, 1000), np.random.normal(60, 10, 1000)]).flatten()
        target = np.array([np.random.normal(10000, 2000, 1000), np.random.normal(8000, 1000, 1000)]).flatten()
        data2 = pd.DataFrame({'first': first, 'second': second})

        reg = DecisionTreeRegressor()
        reg.fit(data2, target)

        self.exp_reg = dx.Explainer(reg, data2, target)
        self.mgf_reg = self.exp_reg.model_fairness(self.protected_reg, 'A')

    def test_fairness_check(self):
        self.mgf.fairness_check()
        self.mgf2.fairness_check()
        self.mgf_reg.fairness_check()

        self.mgf.fairness_check(epsilon=0.1)
        self.mgf_reg.fairness_check(epsilon=0.1)

        with self.assertRaises(dx.fairness._basics.exceptions.ParameterCheckError):
            self.mgf.fairness_check(epsilon=-1)

        with self.assertRaises(dx.fairness._basics.exceptions.ParameterCheckError):
            self.mgf_reg.fairness_check(epsilon=-1)

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

    def test_GroupFairnessRegression(self):
        exp = self.exp_reg
        protected = self.protected_reg

        gfco = dx.fairness.GroupFairnessRegression(y=exp.y,
                                                   y_hat=exp.y_hat,
                                                   protected=protected,
                                                   privileged='A',
                                                   verbose=False,
                                                   label=exp.label)
        self.assertIsInstance(gfco, dx.fairness.GroupFairnessRegression)

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
            gfco = exp.model_fairness(protected=protected,
                                      privileged='male_old',
                                      epsilon=1.1,  # wrong epsilon
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

        exp2 = self.exp_reg
        protected2 = self.protected_reg
        mgf2 = exp2.model_fairness(protected=protected2,
                                 privileged='A',
                                 verbose=False)

        self.assertIsInstance(mgf, dx.fairness.GroupFairnessClassification)
        self.assertIsInstance(mgf2, dx.fairness.GroupFairnessRegression)

    def test_plot_error_handling(self):
        # handling same for all plots, here fairness check plot
        with self.assertRaises(dx.fairness._basics.exceptions.FairnessObjectsDifferenceError):
            mgf_wrong = self.exp.model_fairness(protected=self.german_protected,
                                                privileged='male_young',  # wrong privileged
                                                verbose=False
                                                )
            self.mgf.plot([mgf_wrong])

        with self.assertRaises(TypeError):
            self.mgf.plot(objects= [self.mgf_reg])

        with self.assertRaises(TypeError):
            self.mgf_reg.plot(objects=[self.mgf])

        with self.assertRaises(dx.fairness._basics.exceptions.FairnessObjectsDifferenceError):
            mgf_wrong = self.exp.model_fairness(protected=self.german_protected,
                                                privileged='male_old',
                                                epsilon=0.6,  # different epsilon
                                                verbose=False)
            self.mgf.plot([mgf_wrong])

        with self.assertRaises(dx.fairness._basics.exceptions.ParameterCheckError):
            mgf_wrong = self.exp.model_fairness(protected=self.german_protected,
                                                privileged='male_old',
                                                epsilon=0.8,  # here ok epsilon
                                                verbose=False)
            self.mgf.plot([mgf_wrong], epsilon=1.2)  # here wrong epsilon

        with self.assertRaises(dx.fairness._basics.exceptions.FairnessObjectsDifferenceError):
            exp_wrong = deepcopy(self.exp)
            exp_wrong.y = exp_wrong.y[:-1]
            exp_wrong.y_hat = exp_wrong.y_hat[:-1]

            mgf_wrong = exp_wrong.model_fairness(protected=self.german_protected[:-1],  # shorter protected
                                                 privileged='male_old',
                                                 verbose=False)
            self.mgf.plot([mgf_wrong])

        with self.assertRaises(dx.fairness._basics.exceptions.ParameterCheckError):
            self.mgf.plot(type='not_existing')

    def test_plot_fairness_check(self):
        fig1 = self.mgf.plot(show=False)
        self.assertEqual(fig1.layout.title.text, "Fairness Check")
        self.assertIsInstance(fig1, Figure)

        fig2 = self.mgf.plot(objects=[self.mgf2], show=False)
        self.assertIsInstance(fig2, Figure)

        self.assertEqual(fig2['data'][0]['legendgroup'], "first")
        self.assertEqual(fig2['data'][5]['legendgroup'], "second")

        fig3 = self.mgf_reg.plot(show=False)
        self.assertEqual(fig3.layout.title.text, "Fairness Check")
        self.assertIsInstance(fig3, Figure)

        fig4 = self.mgf.plot(objects=self.mgf2, show=False)
        self.assertIsInstance(fig4, Figure)

        fig5 = self.mgf.plot(objects=self.mgf2, show=False, title='Test title')
        self.assertEqual(fig5.layout.title.text, 'Test title')



    def test_plot_metric_scores(self):
        fig1 = self.mgf.plot(show=False, type='metric_scores')
        self.assertEqual(fig1.layout.title.text, "Metric Scores")
        self.assertIsInstance(fig1, Figure)

        fig2 = self.mgf.plot(objects=[self.mgf2], show=False, type='metric_scores')
        self.assertIsInstance(fig2, Figure)

        self.assertEqual(fig2['data'][0]['legendgroup'], "first")
        self.assertEqual(fig2['data'][5]['legendgroup'], "second")

    def test_plot_radar(self):
        fig1 = self.mgf.plot(show=False, type='radar')
        self.assertEqual(fig1.layout.title.text, "Fairness Radar")
        self.assertIsInstance(fig1, Figure)

        fig2 = self.mgf.plot(objects=[self.mgf2], show=False, type='radar')
        self.assertIsInstance(fig2, Figure)

    def test_plot_heatmap(self):
        fig1 = self.mgf.plot(show=False, type='heatmap')
        self.assertEqual(fig1.layout.title.text, "Fairness Heatmap")
        self.assertIsInstance(fig1, Figure)

        fig2 = self.mgf.plot(objects=[self.mgf2], show=False, type='heatmap')
        self.assertIsInstance(fig2, Figure)

    def test_plot_stacked(self):
        fig1 = self.mgf.plot(show=False, type='stacked')
        self.assertEqual(fig1.layout.title.text, "Stacked Parity Loss Metrics")
        self.assertIsInstance(fig1, Figure)

        fig2 = self.mgf.plot(objects=[self.mgf2], show=False, type='stacked')
        self.assertIsInstance(fig2, Figure)

    def test_plot_performance_and_fairness(self):
        fig1 = self.mgf.plot(show=False, type='performance_and_fairness')
        self.assertEqual(fig1.layout.title.text, "Performance and Fairness")
        self.assertIsInstance(fig1, Figure)

        fig2 = self.mgf.plot(objects=[self.mgf2], show=False, type='performance_and_fairness')
        self.assertIsInstance(fig2, Figure)

    def test_plot_ceteris_paribus_cutoff(self):
        fig1 = self.mgf.plot(show=False, type='ceteris_paribus_cutoff', subgroup='male_old')
        self.assertEqual(fig1.layout.title.text, "Ceteris Paribus Cutoff")
        self.assertIsInstance(fig1, Figure)

        fig2 = self.mgf.plot(objects=[self.mgf2], show=False, type='ceteris_paribus_cutoff', subgroup='male_old')
        self.assertIsInstance(fig2, Figure)

    def test_plot_density(self):
        fig = self.mgf_reg.plot(show=False, type='density')
        self.assertEqual(fig.layout.title.text, "Density plot")

        self.assertIsInstance(fig, Figure)

    def test_mitigation_reweight(self):
        predicted_weights = reweight(np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]), np.array([1, 1, 1, 1, 0, 0, 0, 1, 0, 1]))
        # actual weights as in article
        actual_weights = np.array([0.75, 0.75, 0.75, 0.75, 2, 0.67, 0.67, 1.5, 0.67, 1.5])

        self.assertTrue(np.all(np.around(predicted_weights.astype(float), 2) == actual_weights))

    def test_mitigation_resample(self):
        # uniform
        df = pd.DataFrame({'sex':np.concatenate((np.repeat("M", 5), np.repeat("F", 5), np.repeat("N", 5)), axis = 0),
                           'target': [1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1]})

        MN = sum((df.sex == "M") & (df.target == 0))
        MP = sum((df.sex == "M") & (df.target == 1))
        FN = sum((df.sex == "F") & (df.target == 0))
        FP = sum((df.sex == "F") & (df.target == 1))
        NN = sum((df.sex == "N") & (df.target == 0))
        NP = sum((df.sex == "N") & (df.target == 1))

        weights = reweight(df.sex, df.target)

        wMP = weights[0]
        wMN = weights[4]
        wFP = weights[9]
        wFN = weights[5]
        wNN = weights[12]
        wNP = weights[14]

        # expected
        E_MP = round(MP * wMP)
        E_MN = round(MN * wMN)
        E_FN = round(FN * wFN)
        E_FP = round(FP * wFP)
        E_NP = round(NP * wNP)
        E_NN = round(NN * wNN)

        # uniform
        df_2 = df.iloc[resample(df.sex, df.target), :]

        MN_2 = sum((df_2.sex == "M") & (df_2.target == 0))
        MP_2 = sum((df_2.sex == "M") & (df_2.target == 1))
        FN_2 = sum((df_2.sex == "F") & (df_2.target == 0))
        FP_2 = sum((df_2.sex == "F") & (df_2.target == 1))
        NN_2 = sum((df_2.sex == "N") & (df_2.target == 0))
        NP_2 = sum((df_2.sex == "N") & (df_2.target == 1))

        self.assertEqual(E_MP, MP_2)
        self.assertEqual(E_MN, MN_2)
        self.assertEqual(E_FP, FP_2)
        self.assertEqual(E_FN, FN_2)
        self.assertEqual(E_NP, MP_2)
        self.assertEqual(E_NN, NN_2)

        # preferential

        df = pd.DataFrame({'sex': np.concatenate((np.repeat("M", 5), np.repeat("F", 5), np.repeat("N", 5)), axis=0),
                           'target': [1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                           'name': np.arange(1,16),
                           'probs': [0.9, 0.82, 0.56, 0.78, 0.45, 0.12, 0.48, 0.63, 0.48, 0.88, 0.34, 0.12, 0.34, 0.49, 0.9]})


        df_3 = df.iloc[resample(df.sex, df.target, type = "preferential", probs = df.probs), :]
        self.assertTrue(np.all(sorted(np.array(df_3.name)) == np.array([1, 2, 5, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 15])))

        with self.assertRaises(dx.fairness._basics.exceptions.ParameterCheckError):
            resample(df.sex, df.target, type = "preferential", probs = [0,1,2,3,4])

        with self.assertRaises(dx.fairness._basics.exceptions.ParameterCheckError):
            resample(df.sex, df.target, type = "not_existing", probs = df.probs)

    def test_mitigation_roc(self):
        exp = copy(self.exp)
        exp.y_hat = np.array([0.3, 0.4, 0.45, 0.55, 0.6, 0.7])
        exp.y     = np.array([0, 0, 0, 1, 1, 1])
        protected = ['A', 'A', 'A', 'B', 'B', 'B']
        privileged = 'B'

        exp2 = roc_pivot(exp, protected, privileged, cutoff=0.5, theta=0.09)
        self.assertEqual(list(np.round(exp2.y_hat, 2)), [0.3, 0.4, 0.55, 0.45, 0.6, 0.7])

        exp.y_hat = np.array([0.3, 0.4, 0.45, 0.55, 0.6, 0.7])
        exp.y = np.array([0, 0, 0, 1, 1, 1])
        protected = ['A', 'A', 'A', 'B', 'B', 'B']
        privileged = 'B'

        exp2 = roc_pivot(exp, protected, privileged, cutoff=0.5, theta=0.11)
        self.assertEqual(list(np.round(exp2.y_hat, 2)), [0.3, 0.6, 0.55, 0.45, 0.4, 0.7])

        with self.assertRaises(dx.fairness._basics.exceptions.ParameterCheckError):
            roc_pivot('not explainer', protected, privileged)

        with self.assertRaises(dx.fairness._basics.exceptions.ParameterCheckError):
            roc_pivot(exp, protected, privileged, cutoff=0.5, theta=1.4)

        with self.assertRaises(dx.fairness._basics.exceptions.ParameterCheckError):
            roc_pivot(exp, protected, privileged, cutoff=0.5, theta='sth_else')

if __name__ == '__main__':
    unittest.main()
