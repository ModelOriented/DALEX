import unittest

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import numpy as np
import dalex as dx
from plotly.graph_objs import Figure


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
                              ('classifier', MLPClassifier(hidden_layer_sizes=(50, 100, 50),
                                                           max_iter=400, random_state=0))])

        clf.fit(self.X, self.y)

        self.exp = dx.Explainer(clf, self.X, self.y, verbose=False)

    def test_pdp(self):
        case1 = self.exp.model_profile('partial')
        case2 = self.exp.model_profile('partial', 100, variables=['age', 'fare'])
        case3 = self.exp.model_profile('partial', 120, groups='gender')
        case4 = self.exp.model_profile('partial', 120, groups=['gender'])
        case5 = self.exp.model_profile('partial', 120, groups=np.array(['gender', 'class']))
        case6 = self.exp.model_profile('partial', 120, groups=pd.Series(['gender', 'class']))
        case7 = self.exp.model_profile('partial', 150, variables=['age', 'fare'], groups='class')
        case8 = self.exp.model_profile('partial', 100, variables=['age'], span=0.5, grid_points=30)
        case9 = self.exp.model_profile('partial', None, variables=['age', 'fare', 'gender'], groups=['class', 'embarked'],
                                      span=0.5, grid_points=30)
        case10 = self.exp.model_profile('partial', 100, variables='age', span=0.5, grid_points=30)
        case11 = self.exp.model_profile('partial', 100, variables=np.array(['age', 'class']), span=0.5, grid_points=30)
        case12 = self.exp.model_profile('partial', 100, variables=pd.Series(['age', 'class']), span=0.5, grid_points=30)
        case13 = self.exp.model_profile('partial', 100, intercept=False, span=0.5, grid_points=30)

        fig1 = case1.plot(show=False)
        fig2 = case2.plot(show=False)
        fig3 = case3.plot(show=False)
        fig4 = case4.plot(show=False)
        fig5 = case5.plot(show=False)
        fig6 = case6.plot(show=False)
        fig7 = case7.plot(show=False)
        fig8 = case8.plot(show=False)
        fig9 = case9.plot(show=False)
        fig10 = case10.plot(show=False)
        fig11 = case11.plot(show=False)
        fig12 = case12.plot(show=False)
        fig13 = case13.plot(show=False)

        self.assertIsInstance(case1, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case2, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case3, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case4, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case5, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case6, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case7, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case8, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case9, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case10, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case11, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case12, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case13, dx.dataset_level.AggregatedProfiles)

        self.assertIsInstance(fig1, Figure)
        self.assertIsInstance(fig2, Figure)
        self.assertIsInstance(fig3, Figure)
        self.assertIsInstance(fig4, Figure)
        self.assertIsInstance(fig5, Figure)
        self.assertIsInstance(fig6, Figure)
        self.assertIsInstance(fig7, Figure)
        self.assertIsInstance(fig8, Figure)
        self.assertIsInstance(fig9, Figure)
        self.assertIsInstance(fig10, Figure)
        self.assertIsInstance(fig11, Figure)
        self.assertIsInstance(fig12, Figure)
        self.assertIsInstance(fig13, Figure)

    def test_ale(self):
        case1 = self.exp.model_profile('accumulated')
        case2 = self.exp.model_profile('accumulated', 100, variables=['age', 'fare'])
        case3 = self.exp.model_profile('accumulated', 120, groups='gender')
        case4 = self.exp.model_profile('accumulated', 120, groups=['gender'])
        case5 = self.exp.model_profile('accumulated', 120, groups=np.array(['gender', 'class']))
        case6 = self.exp.model_profile('accumulated', 120, groups=pd.Series(['gender', 'class']))
        case7 = self.exp.model_profile('accumulated', 150, variables=['age', 'fare'], groups='class')
        case8 = self.exp.model_profile('accumulated', 100, variables=['age'], span=0.5, grid_points=30)
        case9 = self.exp.model_profile('accumulated', None, variables=['age', 'fare', 'gender'], groups=['class', 'embarked'],
                                      span=0.5, grid_points=30)
        case10 = self.exp.model_profile('accumulated', 100, variables='age', span=0.5, grid_points=30)
        case11 = self.exp.model_profile('accumulated', 100, variables=np.array(['age', 'class']), span=0.5, grid_points=30)
        case12 = self.exp.model_profile('accumulated', 100, variables=pd.Series(['age', 'class']), span=0.5, grid_points=30)
        case13 = self.exp.model_profile('accumulated', 100, intercept=False, span=0.5, grid_points=30)

        fig1 = case1.plot(show=False)
        fig2 = case2.plot(show=False)
        fig3 = case3.plot(show=False)
        fig4 = case4.plot(show=False)
        fig5 = case5.plot(show=False)
        fig6 = case6.plot(show=False)
        fig7 = case7.plot(show=False)
        fig8 = case8.plot(show=False)
        fig9 = case9.plot(show=False)
        fig10 = case10.plot(show=False)
        fig11 = case11.plot(show=False)
        fig12 = case12.plot(show=False)
        fig13 = case13.plot(show=False)

        self.assertIsInstance(case1, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case2, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case3, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case4, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case5, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case6, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case7, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case8, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case9, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case10, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case11, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case12, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case13, dx.dataset_level.AggregatedProfiles)

        self.assertIsInstance(fig1, Figure)
        self.assertIsInstance(fig2, Figure)
        self.assertIsInstance(fig3, Figure)
        self.assertIsInstance(fig4, Figure)
        self.assertIsInstance(fig5, Figure)
        self.assertIsInstance(fig6, Figure)
        self.assertIsInstance(fig7, Figure)
        self.assertIsInstance(fig8, Figure)
        self.assertIsInstance(fig9, Figure)
        self.assertIsInstance(fig10, Figure)
        self.assertIsInstance(fig11, Figure)
        self.assertIsInstance(fig12, Figure)
        self.assertIsInstance(fig13, Figure)

    def test_conditional(self):
        case1 = self.exp.model_profile('conditional')
        case2 = self.exp.model_profile('conditional', 100, variables=['age', 'fare'])
        case3 = self.exp.model_profile('conditional', 120, groups='gender')
        case4 = self.exp.model_profile('conditional', 120, groups=['gender'])
        case5 = self.exp.model_profile('conditional', 120, groups=np.array(['gender', 'class']))
        case6 = self.exp.model_profile('conditional', 120, groups=pd.Series(['gender', 'class']))
        case7 = self.exp.model_profile('conditional', 150, variables=['age', 'fare'], groups='class')
        case8 = self.exp.model_profile('conditional', 100, variables=['age'], span=0.5, grid_points=30)
        case9 = self.exp.model_profile('conditional', None, variables=['age', 'fare', 'gender'], groups=['class', 'embarked'],
                                      span=0.5, grid_points=30)
        case10 = self.exp.model_profile('conditional', 100, variables='age', span=0.5, grid_points=30)
        case11 = self.exp.model_profile('conditional', 100, variables=np.array(['age', 'class']), span=0.5, grid_points=30)
        case12 = self.exp.model_profile('conditional', 100, variables=pd.Series(['age', 'class']), span=0.5, grid_points=30)
        case13 = self.exp.model_profile('conditional', 100, intercept=False, span=0.5, grid_points=30)

        fig1 = case1.plot(show=False)
        fig2 = case2.plot(show=False)
        fig3 = case3.plot(show=False)
        fig4 = case4.plot(show=False)
        fig5 = case5.plot(show=False)
        fig6 = case6.plot(show=False)
        fig7 = case7.plot(show=False)
        fig8 = case8.plot(show=False)
        fig9 = case9.plot(show=False)
        fig10 = case10.plot(show=False)
        fig11 = case11.plot(show=False)
        fig12 = case12.plot(show=False)
        fig13 = case13.plot(show=False)

        self.assertIsInstance(case1, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case2, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case3, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case4, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case5, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case6, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case7, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case8, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case9, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case10, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case11, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case12, dx.dataset_level.AggregatedProfiles)
        self.assertIsInstance(case13, dx.dataset_level.AggregatedProfiles)

        self.assertIsInstance(fig1, Figure)
        self.assertIsInstance(fig2, Figure)
        self.assertIsInstance(fig3, Figure)
        self.assertIsInstance(fig4, Figure)
        self.assertIsInstance(fig5, Figure)
        self.assertIsInstance(fig6, Figure)
        self.assertIsInstance(fig7, Figure)
        self.assertIsInstance(fig8, Figure)
        self.assertIsInstance(fig9, Figure)
        self.assertIsInstance(fig10, Figure)
        self.assertIsInstance(fig11, Figure)
        self.assertIsInstance(fig12, Figure)
        self.assertIsInstance(fig13, Figure)


if __name__ == '__main__':
    unittest.main()
