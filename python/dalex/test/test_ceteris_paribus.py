import unittest

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import dalex as dx
from dalex.instance_level import CeterisParibus
from dalex.instance_level._ceteris_paribus import utils


class CeterisParibusTestTitanic(unittest.TestCase):
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

        self.exp = dx.Explainer(clf, self.X, self.y)

    def test_calculate_variable_split(self):
        splits = utils.calculate_variable_split(self.X, self.X.columns, 101)
        self.assertIsInstance(splits, (dict,))
        for key, value in splits.items():
            self.assertLessEqual(len(value), 101)

        splits = utils.calculate_variable_split(self.X, ['age', 'country'], 121)
        self.assertIsInstance(splits, (dict,))
        for key, value in splits.items():
            self.assertLessEqual(len(value), 121)

        splits = utils.calculate_variable_split(self.X, ['country'], 5)
        self.assertIsInstance(splits, (dict,))
        for key, value in splits.items():
            self.assertLessEqual(len(value), np.unique(self.X.loc[:, 'country']).shape[0])

    def test_single_variable_profile(self):
        splits = utils.calculate_variable_split(self.X, self.X.columns, 101)
        new_data_age = utils.single_variable_profile(self.exp,
                                                     self.X.iloc[[0], :],
                                                     'age',
                                                     splits['age'])

        new_data_country = utils.single_variable_profile(self.exp,
                                                         self.X.iloc[[0], :],
                                                         'country',
                                                         splits['country'])

        self.assertIsInstance(new_data_age, (pd.DataFrame,))
        self.assertIsInstance(new_data_country, (pd.DataFrame,))

        self.assertLessEqual(new_data_age.shape[0], 101)
        self.assertLessEqual(new_data_country.shape[0], 101)

        self.assertTrue(np.isin(np.array(['_yhat_', '_vname_', '_ids_']),
                                new_data_age.columns).all())

        self.assertTrue(np.isin(np.array(['_yhat_', '_vname_', '_ids_']),
                                new_data_country.columns).all())

        self.assertTrue(np.issubdtype(new_data_age.loc[:, 'age'], np.floating))

    def test_calculate_variable_profile(self):
        splits = utils.calculate_variable_split(self.X, ['age', 'country'], 121)
        vp = utils.calculate_variable_profile(self.exp, self.X.iloc[[0], :], splits)
        self.assertIsInstance(vp, pd.DataFrame)

        splits = utils.calculate_variable_split(self.X, ['country'], 5)
        vp = utils.calculate_variable_profile(self.exp, self.X.iloc[[0], :], splits)
        self.assertIsInstance(vp, pd.DataFrame)

        splits = utils.calculate_variable_split(self.X, self.X.columns, 15)
        vp = utils.calculate_variable_profile(self.exp, self.X.iloc[[0], :], splits)
        self.assertIsInstance(vp, pd.DataFrame)

    def test_calculate_ceteris_paribus(self):
        splits = utils.calculate_variable_split(self.X, ['age', 'country'], 121)

        cp = utils.calculate_ceteris_paribus(self.exp,
                                             self.X.iloc[[0], :].copy(),
                                             splits,
                                             self.y.iloc[0])

        self.assertIsInstance(cp, tuple)
        self.assertIsInstance(cp[0], pd.DataFrame)
        self.assertIsInstance(cp[1], pd.DataFrame)

        splits = utils.calculate_variable_split(self.X, ['country'], 5)

        cp = utils.calculate_ceteris_paribus(self.exp,
                                             self.X.iloc[[0], :].copy(),
                                             splits,
                                             self.y.iloc[0])

        self.assertIsInstance(cp, tuple)
        self.assertIsInstance(cp[0], pd.DataFrame)
        self.assertIsInstance(cp[1], pd.DataFrame)

        splits = utils.calculate_variable_split(self.X, self.X.columns, 15)

        cp = utils.calculate_ceteris_paribus(self.exp,
                                             self.X.iloc[[0], :].copy(),
                                             splits,
                                             self.y.iloc[0])

        self.assertIsInstance(cp, tuple)
        self.assertIsInstance(cp[0], pd.DataFrame)
        self.assertIsInstance(cp[1], pd.DataFrame)

    def test_constructor(self):
        cp = self.exp.predict_profile(self.X.iloc[[0], :])
        self.assertIsInstance(cp, (CeterisParibus,))
        self.assertIsInstance(cp.result, (pd.DataFrame,))
        self.assertIsInstance(cp.new_observation, (pd.DataFrame,))

        with self.assertRaises(ValueError):
            self.exp.predict_profile(self.X.iloc[[0], :], variables=['aaa'])

        with self.assertRaises(TypeError):
            self.exp.predict_profile(self.X.iloc[[0], :], variables='age')

        with self.assertRaises(TypeError):
            self.exp.predict_profile(self.X.iloc[0, :])

        with self.assertRaises(TypeError):
            self.exp.predict_profile(self.X.iloc[[0], :], y=3)


if __name__ == '__main__':
    unittest.main()
