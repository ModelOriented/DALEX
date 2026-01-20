"""
.. include:: ./documentation.md
"""


from . import datasets
from ._explainer.object import Explainer
from .arena.object import Arena
from .aspect import Aspect


__version__ = '1.8.0'

__all__ = [
  "Arena",
  "Aspect",
  "datasets",
  "Explainer",
  "fairness"
]

# specify autocompletion in IPython
# see comment: https://github.com/ska-telescope/katpoint/commit/ed7e8b9e389ee035073c62c2394975fe71031f88
# __dir__ docs (Python 3.7!): https://docs.python.org/3.7/library/functions.html#dir


def __dir__():
    """IPython tab completion seems to respect this."""
    return __all__ + [
        "__all__",
        "__builtins__",
        "__cached__",
        "__doc__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__path__",
        "__spec__",
        "__version__",
    ]
