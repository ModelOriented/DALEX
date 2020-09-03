import numpy as np

def get_variables(explainer):
    indexes = np.apply_along_axis(lambda x, y: not (x == y).all(), 0, explainer.data, explainer.y)
    return explainer.data.columns[indexes]
