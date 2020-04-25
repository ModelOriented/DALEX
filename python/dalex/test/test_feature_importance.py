import unittest

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import dalex as dx
from dalex.dataset_level import VariableImportance
from dalex.dataset_level._variable_importance import utils
from dalex.dataset_level._variable_importance.loss_functions import *


class FeatureImportanceTestTitanic(unittest.TestCase):
    def setUp(self):
        data = pd.read_csv("titanic.csv", index_col=0).dropna()
        data.loc[:, 'survived'] = LabelEncoder().fit_transform(data.survived)

        self.X = data.drop(columns='survived')
        self.y = data.survived

        numeric_features = ['age', 'fare', 'sibsp', 'parch']
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_features = ['gender', 'class', 'embarked', 'country']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', MLPClassifier(hidden_layer_sizes=(150, 100, 50),
                                                           max_iter=500, random_state=0))])

        clf.fit(self.X, self.y)

        self.exp = dx.Explainer(clf, self.X, self.y, verbose=False)

    def test_loss_after_permutation(self):

        variables = {}
        for col in self.X.columns:
            variables[col] = col
        lap = utils.loss_after_permutation(self.exp, loss_root_mean_square,
                                           variables, 100)
        self.assertIsInstance(lap, pd.DataFrame)
        self.assertTrue(np.isin(np.array(['_full_model_', '_baseline_']),
                                lap.columns).all())

        variables = {'age': 'age', 'country': 'country'}
        lap = utils.loss_after_permutation(self.exp, loss_root_mean_square,
                                           variables, 10)
        self.assertIsInstance(lap, pd.DataFrame)
        self.assertTrue(np.isin(np.array(['_full_model_', '_baseline_']),
                                lap.columns).all())

        variables = {'country': 'country'}
        lap = utils.loss_after_permutation(self.exp, loss_root_mean_square,
                                           variables, None)
        self.assertIsInstance(lap, pd.DataFrame)
        self.assertTrue(np.isin(np.array(['_full_model_', '_baseline_']),
                                lap.columns).all())

        variables = {'age': 'age'}
        lap = utils.loss_after_permutation(self.exp, loss_root_mean_square,
                                           variables, None)
        self.assertIsInstance(lap, pd.DataFrame)
        self.assertTrue(np.isin(np.array(['_full_model_', '_baseline_']),
                                lap.columns).all())

    def test_calculate_variable_importance(self):
        variables = {}
        for col in self.X.columns:
            variables[col] = col
        vi = utils.calculate_variable_importance(self.exp,
                                                 'ratio',
                                                 loss_root_mean_square,
                                                 variables,
                                                 100,
                                                 2,
                                                 'aaa',
                                                 True)

        self.assertIsInstance(vi, tuple)
        self.assertIsInstance(vi[0], pd.DataFrame)
        self.assertIsInstance(vi[1], pd.DataFrame)
        self.assertTrue(np.isin(np.array([
            'dropout_loss', 'variable', 'label']),
            vi[0].columns).all())

        vi = utils.calculate_variable_importance(self.exp,
                                                 'difference',
                                                 loss_root_mean_square,
                                                 variables,
                                                 100,
                                                 5,
                                                 'aaa',
                                                 False)

        self.assertIsInstance(vi, tuple)
        self.assertIsInstance(vi[0], pd.DataFrame)
        self.assertIsNone(vi[1])
        self.assertTrue(np.isin(np.array([
            'dropout_loss', 'variable', 'label']),
            vi[0].columns).all())

    def test_constructor(self):
        self.assertIsInstance(self.exp.model_parts(), (VariableImportance,))
        self.assertIsInstance(self.exp.model_parts().result, (pd.DataFrame,))
        self.assertEqual(list(self.exp.model_parts().result.columns),
                         ['variable', 'dropout_loss', 'label'])


if __name__ == '__main__':
    unittest.main()
