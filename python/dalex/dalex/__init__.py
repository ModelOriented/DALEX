# dx.Explainer
# from dalex.datasets import load_*
from . import datasets
from ._explainer.object import Explainer

__version__ = '0.3.0.9000'

__all__ = [
    "Explainer",
    "dataset_level",
    "instance_level",
    "fairness",
    "datasets"
]
