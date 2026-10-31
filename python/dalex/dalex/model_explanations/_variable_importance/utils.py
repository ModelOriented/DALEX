import inspect
import multiprocessing as mp
import warnings
from numpy.random import SeedSequence, default_rng

import numpy as np
import pandas as pd
import narwhals.stable.v2 as nw

from ..._dataframe import take_rows, shuffle_columns, native_namespace, to_pandas
from ..._random import get_random_state


def calculate_variable_importance(explainer,
                                  type,
                                  loss_function,
                                  variables,
                                  N,
                                  B,
                                  label,
                                  processes,
                                  keep_raw_permutations,
                                  random_state,
                                  native=False):
    if processes == 1:
        rng = get_random_state(random_state)
        result = [None] * B
        for i in range(B):
            result[i] = loss_after_permutation(explainer.data, explainer.y, explainer.weights, explainer.model, explainer.predict_function,
                                               loss_function, variables, N, rng, True)
    else:
        # Create number generator for each iteration
        ss = SeedSequence(random_state)
        generators = [default_rng(s) for s in ss.spawn(B)]
        with mp.get_context('spawn').Pool(processes) as pool:
            result = pool.starmap_async(loss_after_permutation, [
                (explainer.data, explainer.y, explainer.weights, explainer.model, explainer.predict_function, loss_function, variables, N, generators[i], True) for
                i in range(B)]).get()

    namespace = native_namespace(explainer.data)
    # Match concat(sort=True)'s feature order, including tied importances.
    portable_names = all(isinstance(name, str) for name in result[0])
    if portable_names:
        names = sorted(result[0])
    else:
        # Arbitrary Python/pandas labels have no common native ordering.
        # Resolve only this small metadata index with the exact historical
        # concat(sort=True) contract; loss reductions remain native below.
        labels = pd.Index(list(result[0]), dtype=object, tupleize_cols=False)
        if isinstance(labels[0], tuple):
            try:
                # Pandas' historical mixed-tuple sort treats strings as
                # sequences too. Pad those metadata sequences explicitly to
                # avoid deprecated ragged-frame construction in pandas 3.
                rows = [list(name) for name in labels]
                width = max(map(len, rows))
                metadata = pd.DataFrame([row + [None] * (width - len(row)) for row in rows])
                names = [labels[i] for i in metadata.sort_values(list(metadata.columns)).index]
            except TypeError:
                # Unorderable Python keys retain their original order, as
                # pandas' safe concat sort does for this compatibility case.
                names = labels.tolist()
        else:
            names = pd.concat([pd.DataFrame(columns=labels)], sort=True).columns.tolist()
    fields = names if portable_names else [f'__loss_{i}' for i in range(len(names))]
    if portable_names:
        loss_columns = {name: [record[name] for record in result] for name in names}
    else:
        # Missing labels are valid pandas metadata, but NaN is not reflexively
        # equal to itself and index normalization can produce a different NaN
        # object. Resolve positions once instead of looking up reconstructed
        # Python keys in each record.
        positions = labels.get_indexer(names)
        records = [list(record.values()) for record in result]
        loss_columns = {field: [record[position] for record in records]
                        for field, position in zip(fields, positions)}
    for field, values in loss_columns.items():
        if any(isinstance(value, np.float32) for value in values):
            # A list of NumPy scalars is inferred differently by dataframe
            # engines (Polars can narrow a mixed list to its first Float32).
            # Resolve numeric promotion before construction, including pandas'
            # historical Float64 promotion when Float32 and None are mixed.
            numeric = np.asarray([np.nan if value is None else value for value in values])
            if numeric.dtype.kind in 'biuf':
                dtype = nw.Float32 if numeric.dtype == np.dtype('float32') else nw.Float64
                loss_columns[field] = nw.new_series(field, values, dtype=dtype, backend=namespace).to_native()
    raw = nw.from_dict(loss_columns, backend=namespace)
    clean_float = (all(dtype == nw.Float64 for dtype in raw.schema.values()) and
                   all(isinstance(value, (float, np.floating)) and not np.isnan(value)
                       for record in result for value in record.values()))
    if clean_float:
        # Standard losses already return float64. Avoid rebuilding a separate
        # conditional/cast expression for every feature when no NaNs need
        # normalization; the same native mean kernel performs the reduction.
        means = raw.select(nw.all().mean()).to_numpy()[0]
    else:
        # Retain the declared stable-v2 minimum and non-float/NaN semantics.
        expressions = []
        for field in fields:
            dtype = nw.Float32 if raw.schema[field] == nw.Float32 else nw.Float64
            column = nw.col(field).cast(dtype)
            expressions.append(nw.when(column.is_nan()).then(None).otherwise(column)
                               .cast(dtype).mean().cast(dtype).alias(field))
        means = raw.select(expressions).to_numpy()[0]
    order = np.argsort(means, kind='quicksort')
    losses = means[order]
    full_model = means[next(i for i, name in enumerate(names) if isinstance(name, str) and name == '_full_model_')]
    with np.errstate(divide='ignore', invalid='ignore'):
        if type == 'ratio':
            losses = losses / full_model
        elif type == 'difference':
            losses = losses - full_model
    if not portable_names:
        # Non-string pandas column labels and arbitrary Python group names are
        # part of the legacy API, but cannot be physical Polars/Arrow names.
        # Keep those labels at the public compatibility boundary; the numerical
        # aggregation above still runs through the input's Narwhals backend.
        aggregated = pd.DataFrame({'variable': [names[i] for i in order],
                                   'dropout_loss': losses, 'label': label})
        permutation = pd.DataFrame({name: loss_columns[field] for name, field in zip(names, fields)})
        return aggregated, permutation if keep_raw_permutations else None
    aggregated = nw.from_dict({'variable': [names[i] for i in order],
                               'dropout_loss': losses, 'label': [label] * len(names)}, backend=namespace).to_native()
    raw_permutations = raw.to_native() if keep_raw_permutations else None
    if native:
        return aggregated, raw_permutations
    return to_pandas(aggregated), to_pandas(raw_permutations) if raw_permutations is not None else None


def loss_after_permutation(data, y, weights, model, predict, loss_function, variables, N, rng, as_dict=False):
    if isinstance(N, int):
        N = min(N, data.shape[0])
        sampled_rows = rng.choice(data.shape[0], N, replace=False)
        sampled_data = take_rows(data, sampled_rows)
        observed = y[sampled_rows]
        sample_weight = weights[sampled_rows] if weights is not None else None
    else:
        sampled_data = data
        observed = y
        sample_weight = weights
    
    # loss on the full model or when outcomes are permuted
    loss_full = calculate_loss(loss_function, observed, predict(model, sampled_data), sample_weight)

    sampled_rows2 = rng.choice(observed.shape[0], observed.shape[0], replace=False)
    sample_weight_rows2 = sample_weight[sampled_rows2] if sample_weight is not None else None
    loss_baseline = calculate_loss(loss_function, observed[sampled_rows2], predict(model, sampled_data), sample_weight_rows2)

    loss_features = {}
    for variables_set_key in variables:
        shuffled_rows = rng.choice(sampled_data.shape[0], sampled_data.shape[0], False)
        ndf = shuffle_columns(sampled_data, variables[variables_set_key], shuffled_rows)
        predicted = predict(model, ndf)

        loss_features[variables_set_key] = calculate_loss(loss_function, observed, predicted, sample_weight)

    loss_features['_full_model_'] = loss_full
    loss_features['_baseline_'] = loss_baseline

    return loss_features if as_dict else pd.DataFrame(loss_features, index=[0])


def calculate_loss(loss_function, observed, predicted, sample_weight=None):
    # Determine if loss function accepts 'sample_weight'
    loss_args = inspect.signature(loss_function).parameters
    supports_weight = "sample_weight" in loss_args

    if supports_weight:
        return loss_function(observed, predicted, sample_weight=sample_weight)
    else:
        if sample_weight is not None:
            raise UserWarning(f"Loss function `{loss_function.__name__}` does not have `sample_weight` argument. Calculating unweighted loss.")
        return loss_function(observed, predicted)
