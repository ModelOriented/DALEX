"""Legacy pandas-returning adapters over shared hierarchical importance."""
from .._table import AspectTable


def calculate_model_hierarchical_importance(aspect, loss_function, type, N, B,
                                            processes, random_state):
    from .._native_triplot import importance_calculator, hierarchical
    importance = importance_calculator(aspect, loss_function=loss_function, kind=type,
        N=N, B=B, processes=processes, random_state=random_state)
    result, full = hierarchical(aspect, importance)
    combined = AspectTable.concatenate(full, columns=full[0]._columns(), keep_index=True)
    return result._dalex_pandas_view(), combined._dalex_pandas_view()


def calculate_single_variable_importance(aspect, loss_function, type, N, B,
                                          processes, random_state):
    from .._native_triplot import importance_calculator, single_variable
    importance = importance_calculator(aspect, loss_function=loss_function, kind=type,
        N=N, B=B, processes=processes, random_state=random_state)
    single, full = single_variable(aspect, importance)
    published = full._dalex_pandas_view()
    published['h'] = 1  # Historical standalone helper published integer height.
    return single._dalex_pandas_view(), published
