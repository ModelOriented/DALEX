"""Shared native hierarchical kernels with explicit legacy publication adapters."""
from ._table import AspectTable
from ._native_dependence import matrix_values
from ._model_aspect_importance.object import ModelAspectImportance
from ._predict_aspect_importance.object import PredictAspectImportance
from .._random import get_random_state


def importance_calculator(aspect, *, loss_function=None, kind='default', N=1000, B=10,
                          processes=1, random_state=None, observation=None, sample_method='default', f=2):
    # The historical hierarchy advanced one serial path stream across cuts and
    # the final single-variable calculation. Resetting at every cut changes
    # nonlinear Shapley estimates despite retaining the same explicit seed.
    rng = get_random_state(random_state) if observation is not None and kind == 'shap' and processes == 1 else None
    def importance(groups):
        if observation is None:
            result = ModelAspectImportance(variable_groups=groups, loss_function=loss_function,
                                            type=kind, N=N, B=B, processes=processes,
                                            random_state=random_state,
                                            _depend_matrix=aspect._native_depend_matrix())
            result.fit(aspect.explainer)
        else:
            result = PredictAspectImportance(variable_groups=groups, type=kind, N=N, B=B,
                                              sample_method=sample_method, f=f, processes=processes,
                                              random_state=random_state,
                                              _depend_matrix=aspect._native_depend_matrix())
            result.fit(aspect.explainer, observation, _rng=rng)
        return result.__dict__['result']
    return importance


def hierarchical(aspect, importance):
    """Calculate every linkage node once, including intermediate tied merges."""
    names = matrix_values(aspect._native_depend_matrix())[1]
    active = {i: [name] for i, name in enumerate(names)}
    order = {name: i for i, name in enumerate(names)}
    results, full = [], []
    for row, (left, right, height, _) in enumerate(aspect.linkage_matrix):
        merged = sorted(active.pop(int(left)) + active.pop(int(right)), key=order.get)
        active[len(names) + row] = merged
        final_at_height = row == len(aspect.linkage_matrix)-1 or aspect.linkage_matrix[row+1, 2] != height
        if final_at_height:
            groups = aspect.get_aspects(1-height)
        else:
            # A flat cut coalesces every merge at this height. Compute each
            # intermediate node as well so a triplot contains all p-1 merges.
            groups, group_number = {}, 1
            for values in sorted(active.values(), key=lambda values: min(order[name] for name in values)):
                key = f'aspect_{group_number}' if len(values) > 1 else values[0]
                groups[key] = values
                group_number += len(values) > 1
        table = importance(groups)
        ids = [i for i, variables in enumerate(table.names) if variables is not None and list(variables) == merged]
        results.append(table.subset(ids))
        if final_at_height:
            full.append(table.with_height(1-height))
    columns = [name for name in results[0]._columns() if name != 'aspect_name']
    result = AspectTable.concatenate(results, columns=columns)
    return result, full


def single_variable(aspect, importance):
    singles = importance(aspect.get_aspects(h=2))
    ids = [i for i, names in enumerate(singles.names) if names is not None]
    columns = ['variable_names', 'dropout_loss', 'dropout_loss_change'] if singles.observation is None else [
        'variable_names', 'variable_values', 'importance']
    single = singles.subset(ids, columns=columns, single=True, keep_index=True)
    return single, singles.with_height(1.)


def calculate(aspect, *, loss_function=None, kind='default', N=1000, B=10,
              processes=1, random_state=None, observation=None, sample_method='default', f=2):
    importance = importance_calculator(aspect, loss_function=loss_function, kind=kind, N=N, B=B,
        processes=processes, random_state=random_state, observation=observation,
        sample_method=sample_method, f=f)
    result, full = hierarchical(aspect, importance)
    single, singles = single_variable(aspect, importance)
    full.append(singles)
    combined = AspectTable.concatenate(full, columns=full[0]._columns(), keep_index=True)
    return result, single, combined
