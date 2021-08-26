import pandas as pd
from copy import deepcopy

from dalex.aspect._model_aspect_importance.object import ModelAspectImportance


def calculate_model_hierarchical_importance(
    aspect,
    loss_function,
    type,
    N,
    B,
    processes,
    random_state
):
    result_df = pd.DataFrame()
    cutting_heights = aspect.linkage_matrix[:, 2]

    aspects_list_previous = aspect.get_aspects(h=2)
    set_aspects_list_previous = {tuple(asp) for asp in aspects_list_previous.values()}
    full_hierarchical_aspect_importance = pd.DataFrame()

    for i in range(len(cutting_heights)):
        aspects_list_current = aspect.get_aspects(1 - cutting_heights[i])
        set_aspects_list_current = {tuple(asp) for asp in aspects_list_current.values()}

        set_diff = set_aspects_list_current - set_aspects_list_previous
        if not bool(set_diff):
            continue
        lastly_merged = list(set_diff)
        lastly_merged = [list(el) for el in lastly_merged]

        current_variable_importance = ModelAspectImportance(
            loss_function=loss_function,
            type=type,
            N=N,
            B=B,
            variable_groups=aspects_list_current, 
            processes=processes,
            random_state=random_state,
            _is_aspect_model_parts=False,
        )

        current_variable_importance.fit(aspect.explainer)

        current_variable_importance_res = current_variable_importance.result
        curr_vi = deepcopy(current_variable_importance_res)
        curr_vi["h"] = 1 - cutting_heights[i]
        full_hierarchical_aspect_importance = pd.concat(
            [full_hierarchical_aspect_importance, curr_vi]
        )

        current_variable_importance_res = current_variable_importance_res[
            ~current_variable_importance_res["aspect_name"].isin(
                ["_baseline_", "_full_model_"]
            )
        ]
        current_variable_importance_res["variables_names"] = (
            current_variable_importance_res["aspect_name"]
            .map(aspects_list_current)
            .apply(lambda x: x.tolist())
        )

        ind = [
            elem in lastly_merged
            for elem in current_variable_importance_res.variables_names
        ]
        lastly_merged_aspect_importance = current_variable_importance_res.loc[ind]

        result_df = pd.concat([result_df, lastly_merged_aspect_importance])
        set_aspects_list_previous = set_aspects_list_current

    result_df = result_df.drop(columns="aspect_name").reset_index(drop=True)
    return result_df, full_hierarchical_aspect_importance
