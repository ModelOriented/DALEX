import unittest

import numpy as np
import pandas as pd

from plotly.graph_objs import Figure

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import dalex as dx


class AggregatedProfilesTestTitanic(unittest.TestCase):
    def setUp(self):
        data = dx.datasets.load_titanic()
        data.loc[:, 'survived'] = LabelEncoder().fit_transform(data.survived)

        self.X = data.drop(columns='survived')
        self.y = data.survived

        numeric_features = ['age', 'fare', 'sibsp', 'parch']
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_features = ['gender', 'class', 'embarked']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', MLPClassifier(hidden_layer_sizes=(20, 20),
                                                           max_iter=400, random_state=0))])
        clf2 = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', MLPClassifier(hidden_layer_sizes=(50, 100, 50),
                                                            max_iter=400, random_state=0))])

        clf.fit(self.X, self.y)
        clf2.fit(self.X, self.y)

        self.exp = dx.Explainer(clf, self.X, self.y, label="model1", verbose=False)
        self.exp2 = dx.Explainer(clf2, self.X, self.y, verbose=False)
        self.exp3 = dx.Explainer(clf, self.X, self.y, label="model3", verbose=False)

    def test_partial(self):
        self.helper_test('partial')

    def test_accumulated(self):
        self.helper_test('accumulated')
        caseCenterNum = self.exp2.model_profile(type='accumulated', center=False)
        caseCenterCat = self.exp2.model_profile(type='accumulated', variable_type='categorical', center=False)

        self.assertIsInstance(caseCenterNum, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(caseCenterCat, dx.dataset_level.AggregatedProfiles)

        figCenterNum = caseCenterNum.plot(show=False)
        figCenterCat = caseCenterCat.plot(show=False)

        self.assertIsInstance(figCenterNum, Figure)
        self.assertIsInstance(figCenterCat, Figure)

        testZerosNum = caseCenterNum.result.groupby('_vname_').apply(lambda x: x['_yhat_'].abs().min()).tolist()
        testZerosCat = caseCenterCat.result.groupby('_vname_').apply(lambda x: x['_yhat_'].abs().min()).tolist()

        self.assertListEqual(testZerosNum, np.zeros(len(testZerosNum)).tolist())
        self.assertListEqual(testZerosCat, np.zeros(len(testZerosCat)).tolist())

    def test_conditional(self):
        self.helper_test('conditional')

    def helper_test(self, test_type):
        case1 = self.exp.model_profile(test_type)
        case2 = self.exp.model_profile(test_type, 100, variables=['age', 'fare'])
        case3 = self.exp.model_profile(test_type, 120, groups='gender')
        case4 = self.exp2.model_profile(test_type, 120, groups=['gender'])
        case5 = self.exp.model_profile(test_type, 120, groups=np.array(['gender', 'class']))
        case6 = self.exp2.model_profile(test_type, 120, groups=pd.Series(['gender', 'class']))
        case7 = self.exp.model_profile(test_type, 150, variables=['age', 'fare'], groups='class')
        case8 = self.exp.model_profile(test_type, 100, variables=['age'], span=0.5, grid_points=30)
        case9 = self.exp2.model_profile(test_type, 100, variables='age', span=0.5, grid_points=30)
        case10 = self.exp.model_profile(test_type, None, variables=['age', 'fare', 'gender'],
                                        groups=['class', 'embarked'],
                                        span=0.5, grid_points=30)
        case11 = self.exp.model_profile(test_type, 100, variables=np.array(['age', 'class']), span=0.5, grid_points=30)
        case12 = self.exp2.model_profile(test_type, 100, variables=pd.Series(['age', 'class']), span=0.5,
                                         grid_points=30)
        case13 = self.exp2.model_profile(test_type, 100, center=False, span=0.5, grid_points=30, processes=2)

        self.assertIsInstance(case1, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case2, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case3, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case4, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case5, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case6, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case7, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case8, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case9, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case10, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case11, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case12, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case13, dx.dataset_level.AggregatedProfiles)

        case_3_models = self.exp3.model_profile(test_type, 100)

        fig1 = case1.plot(show=False, size=3, facet_ncol=1, title="test1", horizontal_spacing=0.2, vertical_spacing=0.2)
        fig2 = case2.plot(show=False)
        fig3 = case3.plot(case4, show=False)
        fig4 = case5.plot(case6, show=False)
        fig5 = case7.plot(show=False)
        fig6 = case8.plot(case9, show=False)
        fig7 = case10.plot(show=False)
        fig8 = case11.plot(case12, show=False)
        fig9 = case13.plot((case1, case_3_models), show=False)

        self.assertIsInstance(fig1, Figure)
        self.assertIsInstance(fig2, Figure)
        self.assertIsInstance(fig3, Figure)
        self.assertIsInstance(fig4, Figure)
        self.assertIsInstance(fig5, Figure)
        self.assertIsInstance(fig6, Figure)
        self.assertIsInstance(fig7, Figure)
        self.assertIsInstance(fig8, Figure)
        self.assertIsInstance(fig9, Figure)


if __name__ == '__main__':
    unittest.main()
