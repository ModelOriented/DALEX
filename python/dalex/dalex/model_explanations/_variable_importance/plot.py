import numpy as np
from types import SimpleNamespace


def texts(frame, rounding_function, digits):
    """Render the final small plot payload without constructing pandas rows."""
    rows = [SimpleNamespace(**row) for row in frame.iter_rows(named=True)]
    labels = [('+' if row.difference > 0 else '') +
              str(rounding_function(np.abs(row.difference), digits)) for row in rows]
    return labels, [tooltip_text(row, rounding_function, digits) for row in rows]


def tooltip_text(row, rounding_function, digits):
    if row.difference > 0:
        key_word = "+"
    else:
        key_word = "-"
    return "Model: " + str(row.label) + " loss after<br>variable: " + str(row.variable) + " is permuted: " + \
           str(rounding_function(row.dropout_loss, digits)) + "<br>" + \
           "Drop-out loss change: " + key_word + str(rounding_function(np.abs(row.difference), digits))
