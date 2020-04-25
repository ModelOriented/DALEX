import unittest

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import dalex as dx


class PredictTestTitanic(unittest.TestCase):
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

    def test_result(self):
        johny_d = pd.DataFrame({'gender': ['male'],
                                'age': [8],
                                'class': ['1st'],
                                'embarked': ['Southampton'],
                                'country': ['England'],
                                'fare': [72],
                                'sibsp': [0],
                                'parch': 0})

        henry = pd.DataFrame({'gender': ['male'],
                              'age': [47],
                              'class': ['1st'],
                              'embarked': ['Cherbourg'],
                              'country': ['United States'],
                              'fare': [25],
                              'sibsp': [0],
                              'parch': [0]})

        self.assertAlmostEqual(self.exp.predict(johny_d)[0], 0.7564213)
        self.assertAlmostEqual(self.exp.predict(henry)[0], 0.15350221)


if __name__ == '__main__':
    unittest.main()
