# dx.Explainer
# from dalex.datasets import load_*
from . import datasets
from ._explainer.object import Explainer

__version__ = '0.2.0.9000'

__all__ = [
    "Explainer",
    "dataset_level",
    "instance_level",
    "datasets",
    "__version__"
]
