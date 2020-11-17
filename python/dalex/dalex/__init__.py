# dx.Explainer
# from dalex.datasets import load_*
from . import datasets
from ._explainer.object import Explainer
from ._arena.object import Arena

__version__ = '0.4.0'

__all__ = [
    "Explainer",
    "dataset_level",
    "instance_level",
    "fairness",
    "datasets",
    "Arena"
]
