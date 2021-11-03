import numpy as np
import pandas as pd
import itertools
from copy import deepcopy
import re

from dalex.aspect._predict_aspect_importance.utils import (
    calculate_predict_aspect_importance,
    calculate_shap_predict_aspect_importance,
)


def calculate_predict_hierarchical_importance(
    aspect,
    new_observation,
    type="default",
    N=2000,
    B=25,
    sample_method="default",
    f=2,
    processes=1,
    random_state=None,
):
    result_df = pd.DataFrame()
    cutting_heights = aspect.linkage_matrix[:, 2]

    aspects_list_previous = aspect.get_aspects(h=2)
    set_aspects_list_previous = {
        tuple(aspect) for aspect in aspects_list_previous.values()
    }

    for i in range(len(cutting_heights)):
        aspects_list_current = aspect.get_aspects(1 - cutting_heights[i])
        set_aspects_list_current = {
            tuple(aspect) for aspect in aspects_list_current.values()
        }

        set_diff = set_aspects_list_current - set_aspects_list_previous
        if not bool(set_diff):
            continue
        lastly_merged = list(set_diff)
        lastly_merged = [list(el) for el in lastly_merged]

        if type == "default":
            current_aspects_importance = calculate_predict_aspect_importance(
                explainer=aspect.explainer,
                new_observation=new_observation,
                variable_groups=aspects_list_current,
                N=N,
                n_aspects=None,
                sample_method=sample_method,
                f=f,
                random_state=random_state,
            )
        else:
            current_aspects_importance = calculate_shap_predict_aspect_importance(
                explainer=aspect.explainer,
                new_observation=new_observation,
                variable_groups=aspects_list_current,
                N=N,
                B=B,
                processes=processes,
                random_state=random_state,
            )

        ind = [
            elem in lastly_merged for elem in current_aspects_importance.variable_names
        ]
        lastly_merged_aspect_importance = current_aspects_importance.loc[ind]

        result_df = pd.concat([result_df, lastly_merged_aspect_importance])
        set_aspects_list_previous = set_aspects_list_current

    result_df = result_df.drop("aspect_name", axis=1).reset_index(drop=True)
    return result_df


def calculate_single_variable_importance(
    aspect,
    new_observation,
    type,
    N,
    B, 
    sample_method,
    f,
    processes,
    random_state
):
    variable_groups = aspect.get_aspects(h=2)
    if type == "default":
        result_df = calculate_predict_aspect_importance(
            aspect.explainer,
            new_observation,
            variable_groups,
            N,
            None,
            sample_method,
            f,
            random_state)
    else:
        result_df = calculate_shap_predict_aspect_importance(
            aspect.explainer,
            new_observation,
            variable_groups,
            N,
            B,
            processes,
            random_state)
    result_df.variable_names = list(
            itertools.chain.from_iterable(result_df.variable_names))
    result_df.variable_values = list(
            itertools.chain.from_iterable(result_df.variable_values))
    result_df = result_df[["variable_names", "variable_values", "importance"]]
    return result_df



def nice_format(x):
    return str(x) if isinstance(x, (str, np.str_)) else str(float(signif(x)))


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
