import unittest

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from dalex.wrappers import ShapWrapper

import dalex as dx


class MLPRegressorTestShapWrapperTitanicFullDataset(unittest.TestCase):
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

        self.exp = dx.Explainer(clf, self.X, self.y, verbose=False)

    def test(self):
        with self.assertRaises(TypeError):
            self.exp.predict_parts(self.X.iloc[[0]], type='shap_wrapper')


class RandomForestClassifierTestShapWrapperTitanicNumericalDataset(unittest.TestCase):
    def setUp(self):
        data = dx.datasets.load_titanic()
        data.loc[:, 'survived'] = LabelEncoder().fit_transform(data.survived)

        self.X = data.loc[:, ["age", "fare", "sibsp", "parch"]]
        self.y = data.survived

        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(self.X, self.y)

        self.exp = dx.Explainer(clf, self.X, self.y, verbose=False)

    def test_predict_parts(self):
        case1 = self.exp.predict_parts(self.X.iloc[[0]], type='shap_wrapper')
        case2 = self.exp.predict_parts(self.X.iloc[1:2, :], type='shap_wrapper', shap_explainer_type='KernelExplainer')
        case3 = self.exp.predict_parts(self.X.iloc[1, :], type='shap_wrapper')

        self.assertIsInstance(case1, ShapWrapper)
        self.assertEqual(case1.shap_explainer_type, 'TreeExplainer')

        self.assertIsInstance(case2, ShapWrapper)
        self.assertEqual(case2.shap_explainer_type, "KernelExplainer")

        self.assertIsInstance(case3, ShapWrapper)

        case1.plot()
        case2.plot()
        case3.plot()

    def test_model_parts(self):
        case1 = self.exp.model_parts(type='shap_wrapper', N=22)
        case2 = self.exp.model_parts(type='shap_wrapper', N=22, shap_explainer_type='KernelExplainer')

        self.assertIsInstance(case1, ShapWrapper)
        self.assertEqual(case1.shap_explainer_type, 'TreeExplainer')

        self.assertIsInstance(case2, ShapWrapper)
        self.assertEqual(case2.shap_explainer_type, "KernelExplainer")

        case1.plot()
        case2.plot()


if __name__ == '__main__':
    unittest.main()
