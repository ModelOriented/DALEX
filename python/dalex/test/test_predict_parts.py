import unittest

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import dalex as dx


class PredictPartsTestTitanic(unittest.TestCase):
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

    def test_bd(self):
        self.assertIsInstance(self.exp.predict_parts(self.X.iloc[[0]], type='break_down'), dx.instance_level.BreakDown)
        self.assertIsInstance(self.exp.predict_parts(self.X.iloc[0], type='break_down'), dx.instance_level.BreakDown)
        self.assertIsInstance(self.exp.predict_parts(self.X.iloc[0].values, type='break_down'), dx.instance_level.BreakDown)
        self.assertIsInstance(self.exp.predict_parts(self.X.iloc[[0]].values, type='break_down'), dx.instance_level.BreakDown)

        with self.assertRaises(ValueError):
            self.exp.predict_parts(self.X.iloc[:2], type='break_down')

        with self.assertRaises(ValueError):
            self.exp.predict_parts(self.X.iloc[:2].values, type='break_down')

        self.assertTrue((self.exp.predict_parts(self.X.iloc[[0]].values, type='break_down',
                               order=['age', 'gender', 'sibsp', 'fare', 'parch', 'class', 'embarked']).result.variable_name.values == ['intercept', 'age', 'gender', 'sibsp', 'fare', 'parch', 'class', 'embarked', '']).all())

        self.assertTrue((self.exp.predict_parts(self.X.iloc[[0]].values, type='break_down',
                                                order=['gender', 'embarked', 'sibsp', 'fare', 'age', 'parch', 'class'
                                                       ]).result.variable_name.values == ['intercept', 'gender', 'embarked',
                                                                                                    'sibsp', 'fare', 'age',
                                                                                                    'parch', 'class', '']).all())

        self.assertTrue((self.exp.predict_parts(self.X.iloc[[0]].values, type='break_down',
                                                order=[3,2,1,0,4,5,6]).result.variable_name.values == ['intercept', 'embarked', 'class', 'age', 'gender', 'fare', 'sibsp', 'parch', '']).all())

        self.assertTrue((self.exp.predict_parts(self.X.iloc[[0]].values, type='break_down',
                                                order=[3, 0, 1, 2, 4, 5, 6]).result.variable_name.values == ['intercept', 'embarked', 'gender', 'age', 'class', 'fare', 'sibsp', 'parch', '']).all())

        self.assertIsInstance(self.exp.predict_parts(self.X.iloc[[0]], type='break_down', path=[3, 0, 1, 2, 4, 5, 6]), dx.instance_level.BreakDown)

        with self.assertRaises(TypeError):
            self.exp.predict_parts(self.X.iloc[[0]], type='break_down', path=['gender', 'embarked', 'sibsp', 'fare', 'age', 'parch', 'class'])

        self.assertIsInstance(self.exp.predict_parts(self.X.iloc[[0]], type='break_down', B=1), dx.instance_level.BreakDown)

        self.assertTrue(hasattr(self.exp.predict_parts(self.X.iloc[[0]], type='break_down', keep_distributions=True), 'yhats_distributions'))

        self.assertIsInstance(self.exp.predict_parts(self.X.iloc[[0]], type='break_down', keep_distributions=True).yhats_distributions,
                                pd.DataFrame)

        # notify?
        self.assertIsInstance(
            self.exp.predict_parts(self.X.iloc[[0]], type='break_down', interaction_preference=2),
            dx.instance_level.BreakDown)

        # notify?
        self.assertIsInstance(
            self.exp.predict_parts(self.X.iloc[[0]], type='break_down', interaction_preference=0.5),
            dx.instance_level.BreakDown)

    def test_ibd(self):
        self.assertIsInstance(self.exp.predict_parts(self.X.iloc[[0]], type='break_down_interactions'), dx.instance_level.BreakDown)
        self.assertIsInstance(self.exp.predict_parts(self.X.iloc[0], type='break_down_interactions'), dx.instance_level.BreakDown)
        self.assertIsInstance(self.exp.predict_parts(self.X.iloc[0].values, type='break_down_interactions'), dx.instance_level.BreakDown)
        self.assertIsInstance(self.exp.predict_parts(self.X.iloc[[0]].values, type='break_down_interactions'), dx.instance_level.BreakDown)

        with self.assertRaises(ValueError):
            self.exp.predict_parts(self.X.iloc[:2], type='break_down_interactions')

        with self.assertRaises(ValueError):
            self.exp.predict_parts(self.X.iloc[:2].values, type='break_down_interactions')

        self.assertTrue((self.exp.predict_parts(self.X.iloc[[0]].values, type='break_down_interactions',
                               order=['age', 'gender', 'sibsp', 'fare', 'parch', 'class', 'embarked']).result.variable_name.values == ['intercept', 'age', 'gender', 'sibsp', 'fare', 'parch', 'class', 'embarked', '']).all())

        self.assertTrue((self.exp.predict_parts(self.X.iloc[[0]].values, type='break_down_interactions',
                                                order=['gender', 'embarked', 'sibsp', 'fare', 'age', 'parch', 'class'
                                                       ]).result.variable_name.values == ['intercept', 'gender', 'embarked',
                                                                                                    'sibsp', 'fare', 'age',
                                                                                                    'parch', 'class', '']).all())

        self.assertTrue((self.exp.predict_parts(self.X.iloc[[0]].values, type='break_down_interactions',
                                                order=[3,2,1,0,4,5,6]).result.variable_name.values == ['intercept', 'embarked', 'class', 'age', 'gender', 'fare', 'sibsp', 'parch', '']).all())

        self.assertTrue((self.exp.predict_parts(self.X.iloc[[0]].values, type='break_down_interactions',
                                                order=[3, 0, 1, 2, 4, 5, 6]).result.variable_name.values == ['intercept', 'embarked', 'gender', 'age', 'class', 'fare', 'sibsp', 'parch', '']).all())

        self.assertIsInstance(self.exp.predict_parts(self.X.iloc[[0]], type='break_down_interactions', path=[3, 0, 1, 2, 4, 5, 6]), dx.instance_level.BreakDown)

        with self.assertRaises(TypeError):
            self.exp.predict_parts(self.X.iloc[[0]], type='break_down_interactions', path=['gender', 'embarked', 'sibsp', 'fare', 'age', 'parch', 'class'])

        self.assertIsInstance(self.exp.predict_parts(self.X.iloc[[0]], type='break_down_interactions', B=1), dx.instance_level.BreakDown)

        self.assertTrue(hasattr(self.exp.predict_parts(self.X.iloc[[0]], type='break_down_interactions', keep_distributions=True),
                                'yhats_distributions'))

        self.assertIsInstance(
            self.exp.predict_parts(self.X.iloc[[0]], type='break_down_interactions', keep_distributions=True).yhats_distributions,
            pd.DataFrame)

        self.assertIsInstance(self.exp.predict_parts(self.X.iloc[[0]], type='break_down_interactions', interaction_preference=2),
                              dx.instance_level.BreakDown)

        self.assertIsInstance(
            self.exp.predict_parts(self.X.iloc[[0]], type='break_down_interactions', interaction_preference=0.5),
            dx.instance_level.BreakDown)

    def test_shap(self):
        self.assertIsInstance(self.exp.predict_parts(self.X.iloc[[0]], type='shap'), dx.instance_level.Shap)
        self.assertIsInstance(self.exp.predict_parts(self.X.iloc[0], type='shap'), dx.instance_level.Shap)
        self.assertIsInstance(self.exp.predict_parts(self.X.iloc[0].values, type='shap'), dx.instance_level.Shap)
        self.assertIsInstance(self.exp.predict_parts(self.X.iloc[[0]].values, type='shap'), dx.instance_level.Shap)

        with self.assertRaises(ValueError):
            self.exp.predict_parts(self.X.iloc[:2], type='shap')

        with self.assertRaises(ValueError):
            self.exp.predict_parts(self.X.iloc[:2].values, type='shap')

        self.assertIsInstance(self.exp.predict_parts(self.X.iloc[[0]].values, type='shap',
                               order=['age', 'gender', 'sibsp', 'fare', 'parch', 'class', 'embarked']), dx.instance_level.Shap)

        self.assertIsInstance(self.exp.predict_parts(self.X.iloc[[0]], type='shap', path=[3, 0, 1, 2, 4, 5, 6]), dx.instance_level.Shap)

        with self.assertRaises(TypeError):
            self.exp.predict_parts(self.X.iloc[[0]], type='shap', path=['gender', 'embarked', 'sibsp', 'fare', 'age', 'parch', 'class'])

        tmp = self.exp.predict_parts(self.X.iloc[[0]].values, type='shap',
                                                path=[3, 0, 1, 2, 4, 5, 6]).result
        self.assertTrue((tmp.loc[tmp.B == 0,'variable_name'].values == [
                            'embarked', 'gender', 'age', 'class', 'fare', 'sibsp', 'parch']).all())

        self.assertIsInstance(self.exp.predict_parts(self.X.iloc[[0]], type='shap', B=1), dx.instance_level.Shap)

        self.assertTrue(hasattr(self.exp.predict_parts(self.X.iloc[[0]], type='shap', keep_distributions=True), 'yhats_distributions'))

        self.assertIsInstance(self.exp.predict_parts(self.X.iloc[[0]], type='shap', keep_distributions=True).yhats_distributions,
                                pd.DataFrame)

        # notify
        self.assertIsInstance(
            self.exp.predict_parts(self.X.iloc[[0]], type='shap', interaction_preference=2),
            dx.instance_level.Shap)

        # notify?
        self.assertIsInstance(
            self.exp.predict_parts(self.X.iloc[[0]], type='shap', interaction_preference=0.5),
            dx.instance_level.Shap)


if __name__ == '__main__':
    unittest.main()
