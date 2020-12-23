import numpy as np


def tooltip_text(row, rounding_function, digits):
    if row.difference > 0:
        key_word = "+"
    else:
        key_word = "-"
    return "Model: " + str(row.label) + " loss after<br>variable: " + str(row.variable) + " is permuted: " + \
           str(rounding_function(row.dropout_loss, digits)) + "<br>" + \
           "Drop-out loss change: " + key_word + str(rounding_function(np.abs(row.difference), digits))
