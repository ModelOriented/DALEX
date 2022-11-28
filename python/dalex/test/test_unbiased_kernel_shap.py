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
        ...


class UnbiasedShapWelford(unittest.TestCase):
    def setUp(self):
        ...

    def test_update(self):
        welford_state = welford.WelfordState()
        welford_state.update(
            np.array(
                [
                    [1, 1, -2],
                    [2, 1, 2],
                    [3, 1, 6],
                ]
            )
        )
        count, b, b_sum_squares = welford_state
        mean, var = welford_state.stats
        self.assertTrue(np.allclose(mean, np.array([2, 1, 2])))
        self.assertTrue(np.allclose(var, np.array([1, 0, 16])))
        self.assertTrue(np.allclose(b_sum_squares, np.array([2, 0, 32])))


if __name__ == "__main__":
    unittest.main()
