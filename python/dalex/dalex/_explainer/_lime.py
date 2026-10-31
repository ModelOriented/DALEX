"""Optional LIME result storage; imported only for a requested LIME explanation."""
from lime.explanation import Explanation as LimeExplanation
from .._explanation import ResultTables


class NativeLimeExplanation(ResultTables, LimeExplanation):
    """A LIME explanation retaining its API and a lazy pandas result view."""
