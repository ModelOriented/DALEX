import unittest

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import dalex as dx
from dalex.dataset_level import ModelPerformance


class ModelPerformanceTestTitanic(unittest.TestCase):
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
                              ('classifier', MLPRegressor(hidden_layer_sizes=(150, 100, 50),
                                                          max_iter=500, random_state=0))])

        clf.fit(self.X, self.y)

        self.exp = dx.Explainer(clf, self.X, self.y)

    def test_constructor(self):
        self.assertIsInstance(self.exp.model_performance('classification'), (ModelPerformance,))
        self.assertIsInstance(self.exp.model_performance('classification').result, (pd.DataFrame,))

        self.assertEqual(self.exp.model_performance('classification').result.shape[0], 1)

    # def test_result(self):
    #    res = self.exp.model_performance('classification').result

    #    self.assertAlmostEqual(res.recall[0], 0.6872246696035242)
    #    self.assertAlmostEqual(res.precision[0], 0.8931297709923665)
    #    self.assertAlmostEqual(res.f1[0], 0.7767634854771786)
    #    self.assertAlmostEqual(res.accuracy[0], 0.8718437351119581)
    #    self.assertAlmostEqual(res.auc[0], 0.9233745280420189)


if __name__ == '__main__':
    unittest.main()
