import unittest

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import dalex as dx
import numpy as np

import warnings


class ExplainerTest(unittest.TestCase):
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
                              ('classifier', MLPRegressor(hidden_layer_sizes=(150, 100, 50),
                                                          max_iter=500, random_state=0))])

        clf.fit(self.X, self.y)
        self.model = clf

    def test(self):
        case1 = dx.Explainer(self.model, self.X, self.y, verbose=False)
        case2 = dx.Explainer(self.model, self.X, None, verbose=False)
        case3 = dx.Explainer(self.model, None, self.y, verbose=False)
        case4 = dx.Explainer(self.model, None, None, verbose=False)

        self.assertIsInstance(case1, dx.Explainer)
        self.assertIsInstance(case2, dx.Explainer)
        self.assertIsInstance(case3, dx.Explainer)
        self.assertIsInstance(case4, dx.Explainer)

        with self.assertRaises(ValueError):
            case2.model_performance()
        with self.assertRaises(ValueError):
            case3.model_parts()
        with self.assertRaises(ValueError):
            case4.model_profile()

        case5 = case2.predict_parts(self.X.iloc[[0]])
        case6 = case2.predict_profile(self.X.iloc[[0]])

        self.assertIsInstance(case5, dx.predict_explanations.BreakDown)
        self.assertIsInstance(case6, dx.predict_explanations.CeterisParibus)

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            case5 = dx.Explainer(self.model, self.X, self.y, predict_function=1, verbose=False)
            assert issubclass(w[-1].category, UserWarning)

        self.assertIsInstance(case5, dx.Explainer)

    def test_errors(self):

        from sklearn.ensemble import RandomForestRegressor

        data = dx.datasets.load_fifa()
        X = data.drop(columns=['nationality', 'value_eur']).iloc[1:100, :]
        y = data['value_eur'][1:100]

        model = RandomForestRegressor()
        model.fit(X, y)

        def predict_function_return_2d(m, d):
            n_rows = d.shape[0]
            prediction = m.predict(d)
            return prediction.reshape((n_rows, 1))

        def predict_function_return_3d(m, d):
            n_rows = d.shape[0]
            prediction = m.predict(d)
            return prediction.reshape((n_rows, 1, 1))

        def predict_function_return_one_element_array(m, d):
            return np.array(0.2)

        warnings.simplefilter("always")
        with warnings.catch_warnings(record=True) as w:
            # Trigger a warning.
            dx.Explainer(model, X, y, verbose=False, model_type='regression',
                         predict_function=predict_function_return_2d)
            assert issubclass(w[-1].category, UserWarning)

        with warnings.catch_warnings(record=True) as w:
            # Trigger a warning.
            dx.Explainer(model, X, y, verbose=False, model_type='regression',
                         predict_function=predict_function_return_3d)
            assert issubclass(w[-1].category, UserWarning)

        with warnings.catch_warnings(record=True) as w:
            # Trigger a warning.
            dx.Explainer(model, X, y, verbose=False, model_type='regression',
                         predict_function=predict_function_return_one_element_array)
            assert issubclass(w[-1].category, UserWarning)
