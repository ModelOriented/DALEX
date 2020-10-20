def intersect_unsorted(values, potential_values):
    """Return an unsorted intersection of two lists"""
    return [val for val in values if val in potential_values]


def convert_float_to_str(x, sign=""):
    """convert floats to str but do better than numpy and pandas"""
    if hasattr(x, 'values'):
        x = x.values
    y = [None]*len(x)
    for i, val in enumerate(x):
        y[i] = sign+str(val) if val > 0 else str(val)
    return y
