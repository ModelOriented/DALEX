def tooltip_text(row, rounding_function, digits, type):
    var_val_string = ""
    for i in range(len(row.variables_names)):
        var_val_string += (
            "<br>" + row.variables_names[i] + " = " + str(row.variables_values[i])
        )
    keyword = "Importance: " if type == "default" else "Contribution: "
    return (
        "Aspect: " + row.aspect_name + "<br>" 
        + f"Min abs depend: {rounding_function(row.min_depend, digits)}<br>" 
        + "(between variables: " + ", ".join(row.vars_min_depend) + ")<br>"
        + keyword + f"{rounding_function(row.importance, digits)}<br>"
        + "Variables:"
        + var_val_string
    )

def get_aspect_importance_colors():
    """return default dalex colors"""
    return ["#8bdcbe", "#f05a71"]
