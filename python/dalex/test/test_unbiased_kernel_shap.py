import unittest

import numpy as np
import pandas as pd
from dalex.predict_explanations._unbiased_kernel_shap import utils


class UnbiasedShapUtils(unittest.TestCase):
    def setUp(self):
        ...

    def test_calculate_A(self):
        A = utils.calculate_A(69)
        self.assertEqual(A.shape, (69, 69))

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

    def test_calculate_exact_result(self):
        ...


if __name__ == "__main__":
    unittest.main()
