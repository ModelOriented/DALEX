import unittest

import numpy as np
from plotly.graph_objs import Figure
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import dalex as dx
from dalex.model_explanations._model_performance.utils import *
from dalex.model_explanations._variable_importance import utils


class FeatureImportanceTestTitanic(unittest.TestCase):
    def setUp(self):
        data = dx.datasets.load_titanic()
        data.loc[:, 'survived'] = LabelEncoder().fit_transform(data.survived)

        self.X = data.drop(columns='survived')
        self.y = data.survived.values

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

    def test_loss_after_permutation(self):

        variables = {}
        for col in self.X.columns:
            variables[col] = col
        lap = utils.loss_after_permutation(self.X, self.y, self.exp.model, self.exp.predict_function, rmse,
                                           variables, 100, np.random)
        self.assertIsInstance(lap, pd.DataFrame)
        self.assertTrue(np.isin(np.array(['_full_model_', '_baseline_']),
                                lap.columns).all(), np.random)

        variables = {'age': 'age', 'embarked': 'embarked'}
        lap = utils.loss_after_permutation(self.X, self.y, self.exp.model, self.exp.predict_function, mad,
                                           variables, 10, np.random)
        self.assertIsInstance(lap, pd.DataFrame)
        self.assertTrue(np.isin(np.array(['_full_model_', '_baseline_']),
                                lap.columns).all())

        variables = {'embarked': 'embarked'}
        lap = utils.loss_after_permutation(self.X, self.y, self.exp.model, self.exp.predict_function, mae,
                                           variables, None, np.random)
        self.assertIsInstance(lap, pd.DataFrame)
        self.assertTrue(np.isin(np.array(['_full_model_', '_baseline_']),
                                lap.columns).all())

        variables = {'age': 'age'}
        lap = utils.loss_after_permutation(self.X, self.y, self.exp.model, self.exp.predict_function, rmse,
                                           variables, None, np.random)
        self.assertIsInstance(lap, pd.DataFrame)
        self.assertTrue(np.isin(np.array(['_full_model_', '_baseline_']),
                                lap.columns).all())

        variables = {'personal': ['gender', 'age', 'sibsp', 'parch'],
                     'wealth': ['class', 'fare']}
        lap = utils.loss_after_permutation(self.X, self.y, self.exp.model, self.exp.predict_function, mae,
                                           variables, None, np.random)
        self.assertIsInstance(lap, pd.DataFrame)
        self.assertTrue(np.isin(np.array(['_full_model_', '_baseline_']),
                                lap.columns).all())

    def test_calculate_variable_importance(self):
        variables = {}
        for col in self.X.columns:
            variables[col] = col
        vi = utils.calculate_variable_importance(self.exp,
                                                 'ratio',
                                                 rmse,
                                                 variables,
                                                 100,
                                                 2,
                                                 'aaa',
                                                 1,
                                                 True,
                                                 None)

        self.assertIsInstance(vi, tuple)
        self.assertIsInstance(vi[0], pd.DataFrame)
        self.assertIsInstance(vi[1], pd.DataFrame)
        self.assertTrue(np.isin(np.array([
            'dropout_loss', 'variable', 'label']),
            vi[0].columns).all())

        vi = utils.calculate_variable_importance(self.exp,
                                                 'difference',
                                                 mae,
                                                 variables,
                                                 100,
                                                 5,
                                                 'aaa',
                                                 2,
                                                 False,
                                                 None)

        self.assertIsInstance(vi, tuple)
        self.assertIsInstance(vi[0], pd.DataFrame)
        self.assertIsNone(vi[1])
        self.assertTrue(np.isin(np.array([
            'dropout_loss', 'variable', 'label']),
            vi[0].columns).all())

    def test_constructor(self):
        case1 = self.exp.model_parts()
        self.assertIsInstance(case1, (dx.model_explanations.VariableImportance,))
        self.assertIsInstance(case1.result, (pd.DataFrame,))
        self.assertEqual(list(case1.result.columns), ['variable', 'dropout_loss', 'label'])

        case2 = self.exp.model_parts(keep_raw_permutations=True)
        self.assertTrue(hasattr(case2, 'permutation'))
        self.assertIsInstance(case2.permutation, pd.DataFrame)

    def test_variables_and_variable_groups(self):

        self.assertIsInstance(self.exp.model_parts(variable_groups={'personal': ['gender', 'age', 'sibsp', 'parch'],
                                                                    'wealth': ['class', 'fare']}),
                              dx.model_explanations.VariableImportance)

        self.assertIsInstance(self.exp.model_parts(variables=['age', 'class']), dx.model_explanations.VariableImportance)

        with self.assertRaises(TypeError):
            self.exp.model_parts(variable_groups=['age'])

        with self.assertRaises(TypeError):
            self.exp.model_parts(variables='age')

        with self.assertRaises(TypeError):
            self.exp.model_parts(variable_groups='age')

        with self.assertRaises(TypeError):
            self.exp.model_parts(variable_groups={'age': 'age'})

        with self.assertRaises(TypeError):
            self.exp.model_parts(variables={'age': 'age'})

        with self.assertRaises(TypeError):
            self.exp.model_parts(variables={'age': ['age']})

        self.assertIsInstance(self.exp.model_parts(variables=['age'], variable_groups={'age': ['age']}),
                              dx.model_explanations.VariableImportance)

    def test_types(self):
        self.assertIsInstance(self.exp.model_parts(type='difference'), dx.model_explanations.VariableImportance)
        self.assertIsInstance(self.exp.model_parts(type='ratio'), dx.model_explanations.VariableImportance)
        self.assertIsInstance(self.exp.model_parts(type='variable_importance'), dx.model_explanations.VariableImportance)
        self.assertIsInstance(self.exp.model_parts(type='feature_importance'), dx.model_explanations.VariableImportance)
        self.assertIsInstance(self.exp.model_parts(type='permutational'), dx.model_explanations.VariableImportance)

        with self.assertRaises(ValueError):
            self.exp.model_parts(type='variable_importancee')

        with self.assertRaises(TypeError):
            self.exp.model_parts(type=['variable_importance'])

    def test_N_and_B(self):
        self.assertIsInstance(self.exp.model_parts(N=100), dx.model_explanations.VariableImportance)
        self.assertIsInstance(self.exp.model_parts(B=1), dx.model_explanations.VariableImportance)

        self.assertIsInstance(self.exp.model_parts(B=2, N=100), dx.model_explanations.VariableImportance)

    def test_loss_functions(self):
        self.assertIsInstance(self.exp.model_parts(loss_function='rmse'), dx.model_explanations.VariableImportance)
        self.assertIsInstance(self.exp.model_parts(loss_function='mae'), dx.model_explanations.VariableImportance)
        self.assertIsInstance(self.exp.model_parts(loss_function='mse'), dx.model_explanations.VariableImportance)
        self.assertIsInstance(self.exp.model_parts(loss_function='mad'), dx.model_explanations.VariableImportance)
        self.assertIsInstance(self.exp.model_parts(loss_function='r2'), dx.model_explanations.VariableImportance)

    def test_parallel(self):
        self.assertIsInstance(self.exp.model_parts(type='difference', processes=2), dx.model_explanations.VariableImportance)
        self.assertIsInstance(self.exp.model_parts(type='ratio', processes=2), dx.model_explanations.VariableImportance)
        self.assertIsInstance(self.exp.model_parts(type='variable_importance', processes=2),
                              dx.model_explanations.VariableImportance)

    def test_plot(self):

        case1 = self.exp.model_parts()
        case2 = self.exp2.model_parts()
        case3 = self.exp3.model_parts()

        self.assertIsInstance(case1, dx.model_explanations.VariableImportance)
        self.assertIsInstance(case2, dx.model_explanations.VariableImportance)

        fig1 = case1.plot((case2, case3), show=False)
        fig2 = case2.plot(case3, split='variable', max_vars=2, show=False)
        fig3 = case1.plot(case2, max_vars=3, digits=2, rounding_function=np.round, show=False)
        fig4 = case2.plot(split='variable', show=False)
        fig5 = case1.plot(case3, bar_width=12, vertical_spacing=0.2, title="title", show=False)

        self.assertIsInstance(fig1, Figure)
        self.assertIsInstance(fig2, Figure)
        self.assertIsInstance(fig3, Figure)
        self.assertIsInstance(fig4, Figure)
        self.assertIsInstance(fig5, Figure)


if __name__ == '__main__':
    unittest.main()
