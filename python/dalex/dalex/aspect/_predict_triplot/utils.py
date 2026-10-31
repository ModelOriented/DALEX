"""Legacy hierarchy adapters and scalar label formatting."""
import numpy as np
from copy import deepcopy
import re


def calculate_predict_hierarchical_importance(aspect, new_observation, type='default',
        N=2000, B=25, sample_method='default', f=2, processes=1, random_state=None):
    from .._native_triplot import importance_calculator, hierarchical
    importance = importance_calculator(aspect, kind=type, N=N, B=B,
        sample_method=sample_method, f=f, processes=processes,
        random_state=random_state, observation=new_observation)
    result, _ = hierarchical(aspect, importance)
    # Dependence metadata was attached by the historical object's fit method,
    # rather than returned by this private intermediate helper.
    result.columns = ['variable_names', 'variable_values', 'importance', 'label']
    return result._dalex_pandas_view()


def calculate_single_variable_importance(aspect, new_observation, type, N, B,
                                        sample_method, f, processes, random_state):
    from .._native_triplot import importance_calculator, single_variable
    importance = importance_calculator(aspect, kind=type, N=N, B=B,
        sample_method=sample_method, f=f, processes=processes,
        random_state=random_state, observation=new_observation)
    result, _ = single_variable(aspect, importance)
    return result._dalex_pandas_view()


def nice_format(x):
    value = np.asarray(x)
    if value.ndim != 0 or value.dtype.kind not in 'biuf':
        return str(x)
    return str(float(signif(x)))


# https://stackoverflow.com/a/59888924
def signif(x, p=4):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


def text_abbreviate(text, max_length, skip_chars="[!@#$=., _^*]", split_char="="):
    if max_length < 1:
        return text
    max_length = int(max_length)
    ## split text to two parts (1st before last split char, second after)
    txt = text.rsplit(split_char, 1)
    # get var_name as 1st part
    var_name = txt[0]

    if len(var_name) <= max_length:
        return text
    # skip skip_chars from var_name
    var_name = re.sub(skip_chars, "", var_name)
    if len(var_name) <= max_length:
        return var_name + "=" + txt[1]
    abbreviate_index = set()
    # get all upper case chars and numbers
    for i, char in enumerate(var_name):
        if char == char.upper():
            abbreviate_index.add(i)
    if len(abbreviate_index) == 0:
        abbreviate_index.add(0)
    uppers_set = deepcopy(abbreviate_index)
    curr_len = len(abbreviate_index)

    if curr_len < max_length:
        i = 1
        while curr_len < max_length:
            for ind in uppers_set:
                if curr_len < max_length:
                    if ind + i not in abbreviate_index:
                        abbreviate_index.add(ind + i)
                        curr_len += 1

            i += 1
    abbreviate = ""
    for ind in sorted(abbreviate_index):
        abbreviate += var_name[ind]
    return abbreviate[:max_length] + " =" + txt[1]
