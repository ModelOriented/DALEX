# dx.Explainer
from ._explainer.object import Explainer

# from dalex.datasets import load_titanic
from . import datasets


__all__ = [
    "Explainer",
    "dataset_level",
    "instance_level",
    "datasets"
]