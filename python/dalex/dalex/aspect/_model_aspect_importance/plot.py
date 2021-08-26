import numpy as np


def tooltip_text(row, rounding_function, digits):
    if row.difference > 0:
        dropout_loss_change_string = "+" + str(
            rounding_function(row.difference, digits)
        )
    else:
        dropout_loss_change_string = str(rounding_function(row.difference, digits))
    return (
        "Aspect: "
        + row.aspect_name
        + "<br>"
        + f"Min abs depend: {rounding_function(row.min_depend, digits)}<br>"
        + "(between variables: "
        + ", ".join(row.vars_min_depend)
        + ")<br>"
        + f"Drop-out loss: {rounding_function(row.dropout_loss, digits)}<br>"
        + "Drop-out loss change: "
        + dropout_loss_change_string
        + "<br>"
        + "Variables:<br>"
        + "<br>".join(row.variable_names)
    )