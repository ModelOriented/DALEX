import unittest

from plotly.graph_objs import Figure
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import dalex as dx

import time
import numpy as np
from dalex._arena.static import get_free_port, try_port
from dalex._arena.plots import *

class ArenaTestTitanic(unittest.TestCase):
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
        self.exp2 = dx.Explainer(clf2, self.X, self.y, label="model2", verbose=False)

        # This plots should be supported
        self.reference_plots = [ROCContainer, ShapleyValuesContainer, BreakDownContainer, CeterisParibusContainer,
            FeatureImportanceContainer, PartialDependenceContainer, AccumulatedDependenceContainer, MetricsContainer]

    def test_supported_plots(self):
        arena = dx.Arena()
        arena.push_model(self.exp)
        arena.push_model(self.exp2)
        plots = arena.get_supported_plots()
        sorting = lambda x: x.__name__
        self.assertEqual(sorted(plots, key=sorting), sorted(self.reference_plots, key=sorting))

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
            raise e

    def test_plots(self):
        arena = dx.Arena()
        arena.push_model(self.exp)
        arena.push_observations(self.X.iloc[[1],])
        arena.fill_cache()
        for p in self.reference_plots:
            ref_counts = list(map(lambda param_type: len(arena.list_params(param_type)), p.info.get('requiredParams')))
            count = np.sum([1 for plot in arena.cache if plot.__class__ == p])
            self.assertEqual(np.prod(ref_counts), count, msg="Count of " + str(p))

if __name__ == '__main__':
    unittest.main()
