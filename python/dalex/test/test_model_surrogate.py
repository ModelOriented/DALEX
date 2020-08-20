import unittest

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

import dalex as dx
import pandas as pd


class PredictSurrogateTestTitanic(unittest.TestCase):
    def setUp(self):
        data = dx.datasets.load_titanic()
        self.X = data.drop(columns=['survived', 'class', 'embarked'])
        self.y = data.survived
        self.X.gender = LabelEncoder().fit_transform(self.X.gender)

        # this checks for no feature_importances_ attribute
        model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=400, random_state=0)
        model.fit(self.X, self.y)
        self.exp = dx.Explainer(model, self.X, self.y, verbose=False)

        data2 = dx.datasets.load_fifa()
        self.X2 = data2.drop(["nationality", "overall", "potential",
                              "value_eur", "wage_eur"], axis=1).iloc[0:2000, 0:10]
        self.y2 = data2['value_eur'].iloc[0:2000]

        # this checks for feature_importances_ attribute
        model2 = RandomForestRegressor(random_state=0)
        model2.fit(self.X2, self.y2)
        self.exp2 = dx.Explainer(model2, self.X2, self.y2, verbose=False)

    def test(self):
        case1 = self.exp.model_surrogate()
        case2 = self.exp2.model_surrogate()
        case3 = self.exp.model_surrogate(max_vars=3, max_depth=2)
        case4 = self.exp2.model_surrogate(max_vars=3, max_depth=2)

        case5 = self.exp.model_surrogate(type='linear')
        case6 = self.exp2.model_surrogate(type='linear')
        case7 = self.exp.model_surrogate(type='linear', max_vars=3)
        case8 = self.exp2.model_surrogate(type='linear', max_vars=3)

        self.assertIsInstance(case1, DecisionTreeClassifier)
        self.assertIsInstance(case1.performance, pd.DataFrame)
        self.assertTrue(hasattr(case1, 'plot'))
        self.assertIsInstance(case2, DecisionTreeRegressor)
        self.assertIsInstance(case2.performance, pd.DataFrame)
        self.assertTrue(hasattr(case2, 'plot'))
        self.assertIsInstance(case3, DecisionTreeClassifier)
        self.assertIsInstance(case3.performance, pd.DataFrame)
        self.assertTrue(hasattr(case3, 'plot'))
        self.assertIsInstance(case4, DecisionTreeRegressor)
        self.assertIsInstance(case4.performance, pd.DataFrame)
        self.assertTrue(hasattr(case4, 'plot'))

        self.assertIsInstance(case5, LogisticRegression)
        self.assertIsInstance(case5.performance, pd.DataFrame)
        self.assertIsInstance(case6, LinearRegression)
        self.assertIsInstance(case6.performance, pd.DataFrame)
        self.assertIsInstance(case7, LogisticRegression)
        self.assertIsInstance(case7.performance, pd.DataFrame)
        self.assertIsInstance(case8, LinearRegression)
        self.assertIsInstance(case8.performance, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
