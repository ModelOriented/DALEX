import unittest
from dalex._global_checks import global_check_import


class Global(unittest.TestCase):

    def setUp(self):
        pass

    def test(self):
        with self.assertRaises(ImportError):
            global_check_import("error_package", "error")
        with self.assertRaises(ImportWarning):
            global_check_import("dalex", "warning")
        global_check_import("shap", "test")
        global_check_import("statsmodels")
        global_check_import("scikit-learn")
        global_check_import("lime")

        with self.assertRaises(ImportError):
            global_check_import("sklearn", "this won't work")
