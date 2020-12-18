import pkg_resources
from importlib import import_module
from re import search
import numpy as np
import pandas as pd 

# WARNING: below code is parsed by setup.py
# WARNING: each dependency should be in new line
# - dalex:          artificial dependency used in test_global.py
# - lime:           Explainer.predict_surrogate
# - scikit-learn:   Explainer.model_surrogate
# - statsmodels     LOWESS trendlines in ResidualDiagnostics.plot
# - shap            ShapWrapper
# - flask:          Arena.run_server
# - flask_cors:     Arena.run_server
# - requests:       Arena.stop_server
OPTIONAL_DEPENDENCIES = {
    'dalex': '99.0',
    'lime': '0.2.0.1',
    'scikit-learn': '0.22.0',
    'statsmodels': '0.11.1',
    'shap': '0.35.0',
    'flask': '1.1.2',
    'flask_cors': '3.0.8',
    'requests': '2.24.0'
}
# WARNING
# WARNING


def global_check_import(name=None, functionality=None):
    # check if the optional dependency is available
    if name:
        try:
            if name == "scikit-learn":
                import_module("sklearn")
            else:
                import_module(name)

            installed_version = pkg_resources.parse_version(pkg_resources.get_distribution(name).version)
            needed_version = pkg_resources.parse_version(OPTIONAL_DEPENDENCIES[name])
            if installed_version < needed_version:
                raise ImportWarning("Outdated version of optional dependency '" + name + "'. " +
                                    ("Update '" + name + "' for " + functionality + ". ") if functionality else "" +
                                    "Use pip or conda to update '" + name + "' to avoid potential errors.")
        except ImportError:
            raise ImportError("Missing optional dependency '" + name + "'. " +
                              ("Install '" + name + "' for " + functionality + ". ") if functionality else "" +
                              "Use pip or conda to install '" + name + "'.")


def global_check_object_class(ob, class_object):
    # check if one of 'objects' is of the desired class - raise error if not
    if not isinstance(ob, class_object):
        class_name = search("(?<=<class ').*(?='>)", str(class_object))[0]
        raise TypeError("One of 'objects' isn't of the " + class_name + " class: " + str(type(ob)))


def global_raise_objects_class(ob, class_object):
    # raise error if 'objects' isn't of the desired class
    class_name = search("(?<=<class ').*(?='>)", str(class_object))[0]
    raise TypeError(
        "'objects' should be either " + class_name +
        " object or list/tuple of " + class_name + " objects: " + str(type(ob))
    )