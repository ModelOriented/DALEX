import unittest

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import numpy as np
import dalex as dx


class APTestTitanic(unittest.TestCase):
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
                              ('classifier', MLPClassifier(hidden_layer_sizes=(150, 100, 50),
                                                           max_iter=500, random_state=0))])

        clf.fit(self.X, self.y)

        self.exp = dx.Explainer(clf, self.X, self.y, verbose=False)

    def test_pdp(self):
        self.assertIsInstance(self.exp.model_profile('partial', 500), dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(self.exp.model_profile('partial', 100, variables=['age', 'fare']), dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(self.exp.model_profile('partial', 120, groups='gender'),
                              dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(self.exp.model_profile('partial', 120, groups=['gender']),
                              dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(self.exp.model_profile('partial', 120, groups=np.array(['gender', 'class'])),
                              dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(self.exp.model_profile('partial', 120, groups=pd.Series(['gender', 'class'])),
                              dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(self.exp.model_profile('partial', 150, variables=['age', 'fare'], groups='class'),
                              dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(self.exp.model_profile('partial', 100, variables=['age'], span=0.5, grid_points=30),
                              dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(self.exp.model_profile('partial', None, variables=['age', 'fare', 'gender'], groups=['class', 'embarked'],
                                                     span=0.5, grid_points=30),
                              dx.dataset_level.AggregatedProfiles)

        self.exp.model_profile('partial', 100, variables='age', span=0.5, grid_points=30)
        self.exp.model_profile('partial', 100, variables=np.array(['age', 'class']), span=0.5, grid_points=30)
        self.exp.model_profile('partial', 100, variables=pd.Series(['age', 'class']), span=0.5, grid_points=30)
        self.exp.model_profile('partial', 100, intercept=False, span=0.5, grid_points=30)

    def test_ale(self):
        self.assertIsInstance(self.exp.model_profile('accumulated', 500), dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(self.exp.model_profile('accumulated', 100, variables=['age', 'fare']), dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(self.exp.model_profile('accumulated', 120, groups='gender'),
                              dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(self.exp.model_profile('accumulated', 120, groups=['gender']),
                              dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(self.exp.model_profile('accumulated', 120, groups=np.array(['gender', 'class'])),
                              dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(self.exp.model_profile('accumulated', 120, groups=pd.Series(['gender', 'class'])),
                              dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(self.exp.model_profile('accumulated', 150, variables=['age', 'fare'], groups='class'),
                              dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(self.exp.model_profile('accumulated', 100, variables=['age'], span=0.5, grid_points=30),
                              dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(self.exp.model_profile('accumulated', None, variables=['age', 'fare', 'gender'], groups=['class', 'embarked'],
                                                     span=0.5, grid_points=30),
                              dx.dataset_level.AggregatedProfiles)

        self.exp.model_profile('accumulated', 100, variables='age', span=0.5, grid_points=30)
        self.exp.model_profile('accumulated', 100, variables=np.array(['age', 'class']), span=0.5, grid_points=30)
        self.exp.model_profile('accumulated', 100, variables=pd.Series(['age', 'class']), span=0.5, grid_points=30)
        self.exp.model_profile('accumulated', 100, intercept=False, span=0.5, grid_points=30)

    def test_conditional(self):
        self.assertIsInstance(self.exp.model_profile('conditional', 500), dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(self.exp.model_profile('conditional', 100, variables=['age', 'fare']), dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(self.exp.model_profile('conditional', 120, groups='gender'),
                              dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(self.exp.model_profile('conditional', 120, groups=['gender']),
                              dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(self.exp.model_profile('conditional', 120, groups=np.array(['gender', 'class'])),
                              dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(self.exp.model_profile('conditional', 120, groups=pd.Series(['gender', 'class'])),
                              dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(self.exp.model_profile('conditional', 150, variables=['age', 'fare'], groups='class'),
                              dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(self.exp.model_profile('conditional', 100, variables=['age'], span=0.5, grid_points=30),
                              dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(self.exp.model_profile('conditional', None, variables=['age', 'fare', 'gender'], groups=['class', 'embarked'],
                                                     span=0.5, grid_points=30),
                              dx.dataset_level.AggregatedProfiles)

        self.exp.model_profile('conditional', 100, variables='age', span=0.5, grid_points=30)
        self.exp.model_profile('conditional', 100, variables=np.array(['age', 'class']), span=0.5, grid_points=30)
        self.exp.model_profile('conditional', 100, variables=pd.Series(['age', 'class']), span=0.5, grid_points=30)
        self.exp.model_profile('conditional', 100, intercept=False, span=0.5, grid_points=30)


if __name__ == '__main__':
    unittest.main()
