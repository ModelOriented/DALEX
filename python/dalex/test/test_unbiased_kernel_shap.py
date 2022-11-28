import unittest

import numpy as np
import pandas as pd
from dalex.predict_explanations._unbiased_kernel_shap import utils, welford


class UnbiasedShapUtils(unittest.TestCase):
    def setUp(self):
        ...

    def test_calculate_A(self):
        A2 = utils.calculate_A(2)
        expected_A2 = np.eye(2) * 0.5
        A3 = utils.calculate_A(3)
        p = 1 / (3 * 2 * (1 / 2 + 1 / 2))
        expected_A3 = np.array(
            [
                [0.5, p, p],
                [p, 0.5, p],
                [p, p, 0.5],
            ]
        )
        self.assertTrue(np.allclose(A2, expected_A2))
        self.assertTrue(np.allclose(A3, expected_A3))

    def test_sample_subsets(self):
        subsets = utils.sample_subsets(1337, 42)
        self.assertEqual(subsets.shape, (1337, 42))

    def test_predict_on_subsets(self):
        feature_subsets = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=bool)
        observation = pd.DataFrame([[1, 2]])
        data = pd.DataFrame([[0, 0], [1, 2], [3, 2], [3, 4], [6, -3]])
        predict_function = lambda arr: arr.max(axis=1)

        subsets_predictions = utils.predict_on_subsets(
            feature_subsets, observation, data, predict_function, len(data)
        )
        expected_output = np.array(
            [np.mean([0, 2, 3, 4, 6]), np.mean([2, 2, 3, 3, 6]), np.mean([1, 2, 2, 4, 1]), 2]
        )

        self.assertEqual(subsets_predictions.shape, (feature_subsets.shape[0],))
        self.assertTrue(np.allclose(subsets_predictions, expected_output))

    def test_calculate_b_sample(self):
        S = np.array(
            [
                [1, 0, 1],
                [1, 1, 0],
            ],
            dtype=bool,
        )
        S_predictions = np.array([2, 3])
        null_prediction = -7
        expected_result = np.array([[9, 0, 9], [10, 10, 0]])
        b_sample = utils.calculate_b_sample(S, S_predictions, null_prediction)
        self.assertTrue(np.allclose(b_sample, expected_result))

    def test_calculate_exact_result(self):
        """let
        model: lambda arr: arr.max(axis=1)
        data: [2, 3], [0, 1]
        our observation: [3, 4]"""
        expected_shap_values = np.array([0.5, 1.5])

        A = utils.calculate_A(2)
        full_prediction = 4
        null_prediction = 2
        S = np.array([[1, 0], [0, 1]], dtype=bool)
        S_predictions = np.array([3, 4])
        b_sample = utils.calculate_b_sample(S, S_predictions, null_prediction)
        b_cov = np.cov(b_sample.T)
        shap_values, std = utils.calculate_exact_result(
            A, b_sample.mean(axis=0), full_prediction, null_prediction, b_cov, 2
        )
        self.assertTrue(np.allclose(expected_shap_values, shap_values))


class UnbiasedShapWelford(unittest.TestCase):
    def setUp(self):
        ...

    def test_update(self):
        welford_state = welford.WelfordState()
        Z = np.array(
            [
                [1, 1, -2],
                [2, 1, 2],
                [3, 1, 6],
            ]
        )
        welford_state.update(Z)
        mean, cov = welford_state.stats
        self.assertTrue(np.allclose(mean, np.array([2, 1, 2])))
        self.assertTrue(np.allclose(cov, np.cov(Z.T)))

    def test_batching(self):
        ws1 = welford.WelfordState()
        ws2 = welford.WelfordState()
        Z = np.random.randn(13, 42)
        ws1.update(Z)
        for z in Z:
            ws2.update(z[None, :])

        mean1, cov1 = ws1.stats
        mean2, cov2 = ws2.stats
        self.assertTrue(np.allclose(mean1, mean2))
        self.assertTrue(np.allclose(cov1, cov2))


if __name__ == "__main__":
    unittest.main()
