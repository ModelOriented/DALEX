import unittest

import dalex as dx
import pandas as pd


class DatasetsTest(unittest.TestCase):
    def setUp(self):
        pass

    def test(self):
        case1 = dx.datasets.load_fifa()
        case2 = dx.datasets.load_titanic()
        case3 = dx.datasets.load_apartments()
        case4 = dx.datasets.load_apartments_test()
        case5 = dx.datasets.load_hr()
        case6 = dx.datasets.load_hr_test()
        case7 = dx.datasets.load_dragons()
        case8 = dx.datasets.load_dragons_test()

        self.assertIsInstance(case1, pd.DataFrame)
        self.assertIsInstance(case2, pd.DataFrame)
        self.assertIsInstance(case3, pd.DataFrame)
        self.assertIsInstance(case4, pd.DataFrame)
        self.assertIsInstance(case5, pd.DataFrame)
        self.assertIsInstance(case6, pd.DataFrame)
        self.assertIsInstance(case7, pd.DataFrame)
        self.assertIsInstance(case8, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
