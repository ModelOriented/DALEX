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
from dalex.instance_level._ceteris_paribus import utils


class CeterisParibusTestTitanic(unittest.TestCase):
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

    def test_calculate_variable_split(self):
        splits = utils.calculate_variable_split(self.X, self.X.columns, 101)
        self.assertIsInstance(splits, (dict,))
        for key, value in splits.items():
            self.assertLessEqual(len(value), 101)

        splits = utils.calculate_variable_split(self.X, ['age', 'fare'], 121)
        self.assertIsInstance(splits, (dict,))
        for key, value in splits.items():
            self.assertLessEqual(len(value), 121)

        splits = utils.calculate_variable_split(self.X, ['gender'], 5)
        self.assertIsInstance(splits, (dict,))
        for key, value in splits.items():
            self.assertLessEqual(len(value), np.unique(self.X.loc[:, 'gender']).shape[0])

    def test_single_variable_profile(self):
        splits = utils.calculate_variable_split(self.X, self.X.columns, 101)
        new_data_age = utils.single_variable_profile(self.exp.predict_function,
                                                     self.exp.model,
                                                     self.X.iloc[[0], :],
                                                     'age',
                                                     splits['age'])

        new_data_embarked = utils.single_variable_profile(self.exp.predict_function,
                                                          self.exp.model,
                                                          self.X.iloc[[0], :],
                                                          'embarked',
                                                          splits['embarked'])

        self.assertIsInstance(new_data_age, (pd.DataFrame,))
        self.assertIsInstance(new_data_embarked, (pd.DataFrame,))

        self.assertLessEqual(new_data_age.shape[0], 101)
        self.assertLessEqual(new_data_embarked.shape[0], 101)

        self.assertTrue(np.isin(np.array(['_yhat_', '_vname_', '_ids_']),
                                new_data_age.columns).all())

        self.assertTrue(np.isin(np.array(['_yhat_', '_vname_', '_ids_']),
                                new_data_embarked.columns).all())

        self.assertTrue(pd.api.types.is_numeric_dtype(new_data_age.loc[:, 'age']))

    def test_calculate_variable_profile(self):
        splits = utils.calculate_variable_split(self.X, ['age', 'gender'], 121)
        vp = utils.calculate_variable_profile(self.exp.predict_function, self.exp.model, self.X.iloc[[0], :], splits, 1)
        self.assertIsInstance(vp, pd.DataFrame)

        splits = utils.calculate_variable_split(self.X, ['gender'], 5)
        vp = utils.calculate_variable_profile(self.exp.predict_function, self.exp.model, self.X.iloc[[0], :], splits, 1)
        self.assertIsInstance(vp, pd.DataFrame)

        splits = utils.calculate_variable_split(self.X, self.X.columns, 15)
        vp = utils.calculate_variable_profile(self.exp.predict_function, self.exp.model, self.X.iloc[[0], :], splits, 2)
        self.assertIsInstance(vp, pd.DataFrame)

    def test_calculate_ceteris_paribus(self):
        splits = utils.calculate_variable_split(self.X, ['age', 'gender'], 121)

        cp = utils.calculate_ceteris_paribus(self.exp,
                                             self.X.iloc[[0], :].copy(),
                                             splits,
                                             self.y.iloc[0],
                                             processes=1,
                                             verbose=False)

        self.assertIsInstance(cp, tuple)
        self.assertIsInstance(cp[0], pd.DataFrame)
        self.assertIsInstance(cp[1], pd.DataFrame)

        splits = utils.calculate_variable_split(self.X, ['embarked'], 5)

        cp = utils.calculate_ceteris_paribus(self.exp,
                                             self.X.iloc[[0], :].copy(),
                                             splits,
                                             self.y.iloc[0],
                                             processes=1,
                                             verbose=False)

        self.assertIsInstance(cp, tuple)
        self.assertIsInstance(cp[0], pd.DataFrame)
        self.assertIsInstance(cp[1], pd.DataFrame)

        splits = utils.calculate_variable_split(self.X, self.X.columns, 15)

        cp = utils.calculate_ceteris_paribus(self.exp,
                                             self.X.iloc[[0], :].copy(),
                                             splits,
                                             self.y.iloc[0],
                                             processes=1,
                                             verbose=False)

        self.assertIsInstance(cp, tuple)
        self.assertIsInstance(cp[0], pd.DataFrame)
        self.assertIsInstance(cp[1], pd.DataFrame)

    def test_constructor(self):
        cp = self.exp.predict_profile(self.X.iloc[[0], :], verbose=False)
        self.assertIsInstance(cp, (dx.instance_level.CeterisParibus,))
        self.assertIsInstance(cp.result, (pd.DataFrame,))
        self.assertIsInstance(cp.new_observation, (pd.DataFrame,))

        with self.assertRaises(ValueError):
            self.exp.predict_profile(self.X.iloc[[0], :], variables=['aaa'], verbose=False)

        with self.assertRaises(TypeError):
            self.exp.predict_profile(self.X.iloc[[0], :], y=3, verbose=False)

        # with self.assertRaises(TypeError):
        #     self.assertIsInstance(self.exp.predict_profile(self.X.iloc[[0], :], variables='age'),
        #                           dx.instance_level.CeterisParibus)

        self.assertIsInstance(self.exp.predict_profile(self.X.iloc[0, :], verbose=False),
                              dx.instance_level.CeterisParibus)
        self.assertIsInstance(self.exp.predict_profile(self.X.iloc[0:10, :], verbose=False),
                              dx.instance_level.CeterisParibus)
        self.assertIsInstance(self.exp.predict_profile(self.X.iloc[[0], :], variables=['age'], verbose=False),
                              dx.instance_level.CeterisParibus)
        self.assertIsInstance(self.exp.predict_profile(self.X.iloc[0, :].values.reshape(-1, ), verbose=False),
                              dx.instance_level.CeterisParibus)
        self.assertIsInstance(self.exp.predict_profile(self.X.iloc[0:10, :].values, verbose=False),
                              dx.instance_level.CeterisParibus)
        self.assertIsInstance(self.exp.predict_profile(self.X.iloc[0:10, :].values, processes=2, verbose=False),
                              dx.instance_level.CeterisParibus)

    def test_plot(self):

        case1 = self.exp.predict_profile(self.X.iloc[2:10, :], verbose=False)
        case2 = self.exp.predict_profile(self.X.iloc[0, :], verbose=False)
        case3 = self.exp.predict_profile(self.X.iloc[1, :], verbose=False)

        self.assertIsInstance(case1, dx.instance_level.CeterisParibus)
        self.assertIsInstance(case2, dx.instance_level.CeterisParibus)

        fig1 = case1.plot((case2, case3), show=False)
        fig2 = case2.plot(variable_type="categorical", show=False)
        fig3 = case1.plot(case2, variables="age", show=False)
        fig4 = case2.plot(variables="gender", show=False)
        fig5 = case1.plot(case3, size=1, color="gender", facet_ncol=1, show_observations=False,
                          title="title", horizontal_spacing=0.2, vertical_spacing=0.2,
                          show=False)
        fig6 = case2.plot(variables=["gender"], show=False)
        fig7 = case2.plot(variables=["gender", 'class'], show=False)
        fig8 = case2.plot(variables=["gender", 'class'], variable_type='categorical', show=False)

        self.assertIsInstance(fig1, Figure)
        self.assertIsInstance(fig2, Figure)
        self.assertIsInstance(fig3, Figure)
        self.assertIsInstance(fig4, Figure)
        self.assertIsInstance(fig5, Figure)
        self.assertIsInstance(fig6, Figure)
        self.assertIsInstance(fig7, Figure)
        self.assertIsInstance(fig8, Figure)


if __name__ == '__main__':
    unittest.main()
