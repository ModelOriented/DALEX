import unittest

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

import dalex as dx
import lime

class PredictSurrogateTestTitanic(unittest.TestCase):
    def setUp(self):
        data = dx.datasets.load_titanic()
        self.X = data.drop(columns=['survived', 'class', 'embarked'])
        self.y = data.survived
        self.X.gender = LabelEncoder().fit_transform(self.X.gender)

        model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=400, random_state=0)
        model.fit(self.X, self.y)
        self.exp = dx.Explainer(model, self.X, self.y, verbose=False)

        data2 = dx.datasets.load_fifa()
        self.X2 = data2.drop(["nationality", "overall", "potential",
                              "value_eur", "wage_eur"], axis=1).iloc[0:2000, 0:10]
        self.y2 = data2['value_eur'].iloc[0:2000]

        model2 = RandomForestRegressor(random_state=0)
        model2.fit(self.X2, self.y2)
        self.exp2 = dx.Explainer(model2, self.X2, self.y2, verbose=False)

    def test(self):
        case1 = self.exp.predict_surrogate(new_observation=self.X.iloc[1, :],
                                           feature_names=self.X.columns)
        # error for num_features=K and mode='classification'
        # case2 = self.exp.predict_surrogate(new_observation=self.X.iloc[1:2, :],
        #                                    mode='classification',
        #                                    feature_names=self.X.columns,
        #                                    discretize_continuous=True,
        #                                    num_features=4)
        case3 = self.exp.predict_surrogate(new_observation=self.X.iloc[1:2, :].to_numpy(),
                                           feature_names=self.X.columns,
                                           kernel_width=2,
                                           num_samples=50)
        case4 = self.exp2.predict_surrogate(new_observation=self.X2.iloc[1, :],
                                            feature_names=self.X2.columns)
        case5 = self.exp2.predict_surrogate(new_observation=self.X2.iloc[1:2, :],
                                            mode='regression',
                                            feature_names=self.X2.columns,
                                            discretize_continuous=True,
                                            num_features=4)
        case6 = self.exp2.predict_surrogate(new_observation=self.X2.iloc[1:2, :].to_numpy(),
                                            feature_names=self.X2.columns,
                                            kernel_width=2,
                                            num_samples=50)

        self.assertIsInstance(case1, lime.explanation.Explanation)
        # self.assertIsInstance(case2, lime.explanation.Explanation)
        self.assertIsInstance(case3, lime.explanation.Explanation)
        self.assertIsInstance(case4, lime.explanation.Explanation)
        self.assertIsInstance(case5, lime.explanation.Explanation)
        self.assertIsInstance(case6, lime.explanation.Explanation)


if __name__ == '__main__':
    unittest.main()
