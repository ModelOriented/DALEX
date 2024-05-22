import unittest
import sys

from plotly.graph_objs import Figure
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import dalex as dx

import time
import numpy as np
import pandas as pd
from dalex.arena.static import get_free_port, try_port
from dalex.arena.plots import *

class ArenaTestTitanic(unittest.TestCase):
    def setUp(self):
        data = dx.datasets.load_titanic()
        data.loc[:, 'survived'] = LabelEncoder().fit_transform(data.survived)

        self.X = data.drop(columns='survived')
        self.y = data.survived
        self.john = pd.DataFrame({'gender': ['male'], 'age': [25], 'class': ['1st'], 'embarked': ['Southampton'],
                                   'fare': [72], 'sibsp': [0], 'parch': 0}, index = ['John'])

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
        self.exp2 = dx.Explainer(clf2, self.X, self.y, label="model2", verbose=False)

        # This plots should be supported
        self.reference_plots = [ROCContainer, ShapleyValuesContainer, BreakDownContainer, CeterisParibusContainer,
            FeatureImportanceContainer, PartialDependenceContainer, AccumulatedDependenceContainer, MetricsContainer, 
            FairnessCheckContainer, ShapleyValuesDependenceContainer, ShapleyValuesVariableImportanceContainer,
            VariableAgainstAnotherContainer, VariableDistributionContainer]

    @unittest.skipIf(sys.platform.startswith("win"), "requires Windows")
    def test_supported_plots(self):
        arena = dx.Arena()
        arena.push_model(self.exp)
        arena.push_model(self.exp2)
        plots = arena.get_supported_plots()
        sorting = lambda x: x.__name__
        self.assertEqual(sorted(plots, key=sorting), sorted(self.reference_plots, key=sorting))

        try:
            arena.stop_server()
        except Exception:
            pass

    @unittest.skipIf(sys.platform.startswith("win"), "requires Windows")
    def test_server(self):
        arena = dx.Arena()
        arena.push_model(self.exp)
        arena.push_model(self.exp2)
        port = get_free_port()
        try:
            arena.run_server(port=port)
            time.sleep(2)
            self.assertFalse(try_port(port))
            arena.stop_server()
        except AssertionError as e:
            arena.stop_server()

        try:
            arena.stop_server()
        except Exception:
            pass

    @unittest.skipIf(sys.platform.startswith("win"), "requires Windows")
    def test_plots(self):
        arena = dx.Arena()
        arena.push_model(self.exp)
        arena.push_observations(self.X.iloc[[1],])
        arena.set_option('DatasetShapleyValues', 'N', 10)
        arena.plots_manager.fill_cache()
        for p in self.reference_plots:
            ref_counts = list(map(lambda param_type: len(arena.list_params(param_type)), p.info.get('requiredParams')))
            count = np.sum([1 for plot in arena.plots_manager.cache if plot.__class__ == p])
            self.assertEqual(np.prod(ref_counts), count, msg="Count of " + str(p))

        try:
            arena.stop_server()
        except Exception:
            pass

    @unittest.skipIf(sys.platform.startswith("win"), "requires Windows")
    def test_observation_attributes(self):
        arena = dx.Arena()
        arena.push_model(self.exp)
        titanic = dx.datasets.load_titanic().iloc[1:20]
        arena.push_observations(titanic) #with m2_price
        arena.push_observations(self.john) #without m2_price
        attrs = arena.get_params_attributes('observation')
        attrs_name = list(map(lambda x: x.get('name'), attrs))
        self.assertEqual(sorted(attrs_name), sorted(titanic.columns))
        for attr in attrs:
            self.assertTrue(all(attr.get('values')[:-1] == titanic[attr.get('name')]))

        try:
            arena.stop_server()
        except Exception:
            pass
        
    @unittest.skipIf(sys.platform.startswith("win"), "requires Windows")
    def test_variable_attributes(self):
        arena = dx.Arena()
        arena.push_model(self.exp)
        titanic = dx.datasets.load_titanic().iloc[1:20]
        arena.push_observations(titanic)
        attrs = { k: arena.get_param_attributes('variable', k) for k in self.X.columns }
        self.assertEqual(attrs, {
            'age': {'type': 'numeric', 'min': 0.1666666667, 'max': 74.0, 'levels': None},
            'class': {'type': 'categorical', 'min': None, 'max': None, 'levels': ['1st', '2nd', '3rd', 'deck crew', 'engineering crew', 'restaurant staff', 'victualling crew']},
            'embarked': {'type': 'categorical', 'min': None, 'max': None, 'levels': ['Belfast', 'Cherbourg', 'Queenstown', 'Southampton']},
            'fare': {'type': 'numeric', 'min': 0.0, 'max': 512.0607, 'levels': None},
            'gender': {'type': 'categorical', 'min': None, 'max': None, 'levels': ['female', 'male']},
            'parch': {'type': 'numeric', 'min': 0, 'max': 9, 'levels': [0, 1, 2, 3, 4, 5, 6, 9]},
            'sibsp': {'type': 'numeric', 'min': 0, 'max': 8, 'levels': [0, 1, 2, 3, 4, 5, 8]}
        })

        try:
            arena.stop_server()
        except Exception:
            pass

if __name__ == '__main__':
    unittest.main()
