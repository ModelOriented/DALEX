import unittest

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import dalex as dx
from plotly.graph_objs import Figure


class ModelPerformanceTestTitanic(unittest.TestCase):
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
                              ('classifier', MLPClassifier(hidden_layer_sizes=(50, 100, 50),
                                                           max_iter=400, random_state=0))])

        clf.fit(self.X, self.y)

        self.exp = dx.Explainer(clf, self.X, self.y, verbose=False)
        self.exp2 = dx.Explainer(clf, self.X, self.y, label="model2", verbose=False)

    def test_constructor(self):
        case1 = self.exp.model_performance('classification')
        self.assertIsInstance(case1, (dx.dataset_level.ModelPerformance,))
        self.assertIsInstance(case1.result, (pd.DataFrame,))
        self.assertEqual(case1.result.shape[0], 1)
        self.assertTrue(np.isin(['recall', 'precision', 'f1', 'accuracy', 'auc'], case1.result.columns).all())

        case2 = self.exp.model_performance('regression')
        self.assertIsInstance(case2, (dx.dataset_level.ModelPerformance,))
        self.assertIsInstance(case2.result, (pd.DataFrame,))
        self.assertEqual(case2.result.shape[0], 1)
        self.assertTrue(np.isin(['mse', 'rmse', 'r2', 'mae', 'mad'], case2.result.columns).all())

    def test_plot(self):

        case1 = self.exp.model_performance('classification')
        case2 = self.exp2.model_performance('classification')

        self.assertIsInstance(case1, dx.dataset_level.ModelPerformance)
        self.assertIsInstance(case2, dx.dataset_level.ModelPerformance)

        fig1 = case1.plot(title="test1", show=False)
        fig2 = case2.plot(case1, show=False)

        self.assertIsInstance(fig1, Figure)
        self.assertIsInstance(fig2, Figure)


if __name__ == '__main__':
    unittest.main()
