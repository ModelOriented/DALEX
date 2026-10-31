"""Compare private pandas helper contracts and index-sensitive public callbacks.

The reference package is unmodified and imported in a separate interpreter.
Pickles retain indexes, extension dtypes and heterogeneous metadata cells.
"""
import argparse
import os
from pathlib import Path
import pickle
import subprocess
import sys
import json
from contextlib import redirect_stdout
from io import StringIO


def indexed_predict(model, frame):
    return frame.to_numpy().sum(axis=1) + frame.index.to_numpy() / 10


def residual(model, frame, target):
    return target - model.predict(frame)


def indexed_residual(model, frame, target):
    return target - indexed_predict(model, frame)


def matrix_predict(model, frame):
    return model.predict(frame.to_numpy())


def matrix_class_predict(model, frame):
    return model.predict_proba(frame.to_numpy())[:, 1]


def float32_loss(observed, predicted):
    import numpy as np
    return np.float32(np.mean((observed - predicted)**2))


def mixed_precision_loss(observed, predicted):
    import numpy as np
    value = np.mean((observed - predicted)**2)
    return np.float32(value) if value < 1 else np.float64(value)


def float32_missing_loss(observed, predicted):
    import numpy as np
    return np.float32(np.nan) if observed[0] > 0 else float32_loss(observed, predicted)


def capture(root, labels_only=False):
    sys.path.insert(0, str(Path(root).resolve()))
    import numpy as np
    import pandas as pd
    import dalex as dx
    from sklearn.linear_model import LinearRegression
    from dalex.predict_explanations._break_down import utils as bd
    from dalex.predict_explanations._shap import utils as shap
    from dalex.predict_explanations._ceteris_paribus import utils as cp
    from dalex.model_explanations._aggregated_profiles import utils as ap
    from dalex.aspect._predict_aspect_importance import utils as pai
    from dalex.aspect._predict_triplot import utils as pt
    from dalex.aspect._model_triplot import utils as mt
    from dalex.aspect import utils as dependence
    if not hasattr(np, 'in1d'):
        np.in1d = np.isin
    rng = np.random.default_rng(682)
    data = pd.DataFrame(rng.normal(size=(32, 3)), columns=['z', 'a', 'b'], index=np.arange(32) * 3 + 7)
    y = data.z.to_numpy() * 2 + data.a.to_numpy() / 3 + np.sin(data.b.to_numpy())
    model = LinearRegression().fit(data, y)
    exp = dx.Explainer(model, data, y, residual_function=residual, label='model', verbose=False)
    observation = data.iloc[[2]]
    result = {}

    def save(name, value):
        if isinstance(value, tuple):
            for index, item in enumerate(value):
                save(f'{name}/{index}', item)
        else:
            result[name] = value

    # Non-string physical pandas labels and partial MultiIndex selectors are
    # real legacy workflows. Native engines cannot use them as physical column
    # names, so their small publication/label-order boundary stays explicit.
    multi = data.copy()
    multi.columns = pd.MultiIndex.from_tuples([('g', 1), ('g', 2), ('h', 1)], names=['group', 'feature'])
    integer = data.copy()
    integer.columns = [10, 20, 30]
    missing_label = data.copy()
    missing_label.columns = pd.Index([None, 'a', 'b'], dtype=object)
    heterogeneous = {pd.Timestamp('2020-01-01'): 'z', 7: 'a', None: 'b'}
    labeled_cases = [('multiindex', multi, None), ('multiindex-prefix', multi, ['g']),
                     ('integer', integer, None), ('none-label', missing_label, None),
                     ('python-group-labels', data, heterogeneous)]
    deep = data.copy()
    deep.columns = pd.MultiIndex.from_tuples([('level',) * 13 + (2**63 + offset,) for offset in [2, 1, 3]])
    labeled_cases.append(('deep-integer-levels', deep, None))
    for case, label in [('nan-label', np.nan), ('na-label', pd.NA), ('nat-label', pd.NaT)]:
        frame = data.copy()
        frame.columns = pd.Index([label, 'a', 'b'], dtype=object)
        labeled_cases.append((case, frame, None))
    for case, frame, variables in labeled_cases:
        labeled = dx.Explainer(model, frame, y, predict_function=indexed_predict,
            residual_function=indexed_residual, label='labeled', verbose=False)
        for kind in ['variable_importance', 'ratio', 'difference']:
            options = ({'variable_groups': {name: [feature] for name, feature in variables.items()}}
                       if isinstance(variables, dict) else {'variables': variables})
            value = labeled.model_parts(**options, type=kind, N=16, B=3, random_state=42)
            save(f'labels-{case}-{kind}', (value.result, value.permutation))
    for case, labels in [('integer', [10, 20, 30]), ('mixed', ['x', 20, ('q', 1)]),
                          ('empty-string', ['', 'x', 'z'])]:
        frame = data.copy()
        frame.columns = labels
        labeled = dx.Explainer(model, frame, y, predict_function=indexed_predict,
            residual_function=indexed_residual, label='labeled', verbose=False)
        save(f'labels-{case}-cp', labeled.predict_profile(frame.head(1), variables=[labels[0]],
            grid_points=3, verbose=False).result)
        for kind in ['partial', 'conditional', 'accumulated']:
            save(f'labels-{case}-{kind}', labeled.model_profile(type=kind, variables=[labels[0]],
                N=16, grid_points=3, random_state=42, verbose=False).result)
    # Narwhals forbids duplicate physical names, but these three pandas APIs
    # worked upstream. Compare their actual outputs rather than introducing a
    # new global uniqueness requirement for original pandas features.
    duplicates = [
        ('strings', ['x', 'x', 'z']), ('integers', [10, 10, 20]),
        ('missing', [None, None, 'z']), ('reserved', ['y_hat', 'y_hat', 'label']),
        ('multiindex', pd.MultiIndex.from_tuples([('g', 1), ('g', 1), ('h', 1)])),
        ('reserved-multiindex', pd.MultiIndex.from_tuples([('y_hat', 1), ('y_hat', 1), ('label', 1)])),
    ]
    for case, names in duplicates:
        frame = data.copy()
        frame.columns = names
        labeled = dx.Explainer(model, frame, y, predict_function=indexed_predict,
            residual_function=indexed_residual, label='labeled', verbose=False)
        for kind in ['variable_importance', 'ratio', 'difference']:
            value = labeled.model_parts(type=kind, N=16, B=3, random_state=42)
            save(f'labels-duplicate-{case}-{kind}', (value.result, value.permutation))
        save(f'labels-duplicate-{case}-performance', labeled.model_performance().result)
        for selection in [None, [frame.columns[0]], [frame.columns[2]]]:
            save(f'labels-duplicate-{case}-diagnostics-{selection}',
                 labeled.model_diagnostics(variables=selection).result)
        if case == 'strings':
            value = labeled.model_parts(N=16, B=3, random_state=42, processes=2)
            save('labels-duplicate-processes', (value.result, value.permutation))
            lime_exp = dx.Explainer(model, frame.copy(), y, predict_function=matrix_predict,
                                   model_type='regression', label='duplicate-lime', verbose=False)
            save('labels-duplicate-lime', lime_exp.predict_surrogate(frame.iloc[0].to_numpy(),
                 random_state=42, num_samples=128, num_features=3).result)
    if labels_only:
        return result
    duplicate_frame = data.copy()
    duplicate_frame.columns = ['x', 'x', 'z']
    regression = dx.Explainer(model, duplicate_frame, y, predict_function=matrix_predict,
                              model_type='regression', label='duplicate-fairness', verbose=False)
    protected = np.array(['u', 'v'] * 16)
    save('duplicate-regression-fairness', regression.model_fairness(protected, 'u').result)
    from sklearn.linear_model import LogisticRegression
    binary_y = (y > np.median(y)).astype(int)
    binary_model = LogisticRegression(random_state=42).fit(data.to_numpy(), binary_y)
    classification = dx.Explainer(binary_model, duplicate_frame, binary_y,
        predict_function=matrix_class_predict, model_type='classification', label='duplicate-fairness', verbose=False)
    fairness = classification.model_fairness(protected, 'u')
    save('duplicate-classification-fairness', (fairness.result, fairness.metric_scores, fairness.parity_loss))
    for loss in [float32_loss, mixed_precision_loss, float32_missing_loss]:
        for kind in ['variable_importance', 'ratio', 'difference']:
            value = exp.model_parts(loss_function=loss, type=kind, N=16, B=7, random_state=42)
            save(f'loss-dtype-{loss.__name__}-{kind}', (value.result, value.permutation))

    mixed = data.assign(cat1=['low', 'high'] * 16, cat2=['yes', 'no', 'no', 'yes'] * 8)
    save('categorical-association', dependence.calculate_cat_assoc_matrix(mixed, ['cat2', 'cat1'], len(mixed)))
    save('categorical-numeric-association', dependence.calculate_cat_num_assoc_matrix(
        mixed, ['cat2', 'cat1'], ['a', 'z'], len(mixed)))

    splits = {'z': np.array([-1., 0., 1.]), 'a': np.array([-.5, .5])}
    save('cp-single', cp.single_variable_profile(exp.predict_function, exp.model, data.iloc[:3], 'z', splits['z']))
    save('cp-many', cp.calculate_variable_profile(exp.predict_function, exp.model, data.iloc[:3], splits, 1, False))
    save('cp-both', cp.calculate_ceteris_paribus(exp, data.iloc[:3], splits, y[:3], 1, False))
    for kind in ['1d', '2d']:
        for order in [None, np.array([2, 0, 1]), np.array(['b', 'z', 'a'])]:
            for distributions in [False, True]:
                save(f'bd-{kind}-{order}-{distributions}', bd.local_interactions(
                    exp, observation, 1., kind, order, distributions))
    one = bd.calculate_1d_changes(exp, observation)
    save('bd-one', one)
    indices = pd.DataFrame({'ind1': [1, 2, 2], 'ind2': [0, 0, 1]})
    save('bd-two', bd.calculate_2d_changes(exp, observation, indices, one - exp.y_hat.mean()))
    feature_path = pd.DataFrame({'diff': one, 'ind1': np.arange(3)})
    for order in [None, np.array([2, 0, 1]), np.array(['b', 'z', 'a'])]:
        ordered = bd.create_ordered_path(feature_path, one.index, '1d', order)
        save(f'bd-order-{order}', ordered)
        save(f'bd-contributions-{order}', bd.calculate_contributions_along_path(
            exp, observation, ordered, exp.y_hat.mean(), exp.predict(observation), '1d', True))
    tied = pd.DataFrame({'adiff_norm': [1., 2., 1., 2., 1., 2.] * 5}, index=[f'x{i}' for i in range(30)])
    save('bd-floating-tie-order', bd.create_ordered_path(tied, tied.index, '2d', None))
    for path in [None, 'average', np.array([2, 0, 1])]:
        for distributions in [False, True]:
            np.random.seed(42)
            save(f'shap-{path}-{distributions}', shap.shap(exp, observation, path, distributions, 3, 1, 42))
    save('shap-processes', shap.shap(exp, observation, 'average', False, 3, 2, 42))
    save('shap-single', shap.get_single_random_path(exp.predict_function, exp.model, data, exp.label,
                                                  observation, np.array([2, 0, 1]), 7))
    save('shap-iterate', shap.iterate_paths(exp.predict_function, exp.model, data, exp.label,
                                         observation, 3, 7, np.random.RandomState(42)))
    save('shap-distributions', shap.calculate_yhats_distributions(exp))
    profiles = cp.calculate_variable_profile(exp.predict_function, exp.model, data.iloc[:5], splits, 1, False)
    profiles['_label_'] = exp.label
    profiles, variables = ap.prepare_numerical_categorical(profiles, None, 'numerical')
    save('ap-variables', variables)
    profiles = ap.prepare_x(profiles, 'numerical')
    save('ap-prepared', profiles)
    profiles['group'] = np.where(profiles['_ids_'].to_numpy() % 2 == 0, 'even', 'odd')
    for kind in ['partial', 'conditional', 'accumulated']:
        for groups in [[], ['group']]:
            save(f'ap-{kind}-{groups}', ap.aggregate_profiles(profiles.copy(), exp.y_hat.mean(), kind,
                                                             groups, True, .25, False))
            save(f'ap-split-{kind}-{groups}', ap.split_over_variables_and_labels(
                profiles.loc[profiles['_vname_'] == 'z'].copy(), kind, groups, .25))
    groups = {'za': np.array(['z', 'a']), 'b': np.array(['b'])}
    for count in [None, 1]:
        for sampling in ['default', 'binom']:
            save(f'aspect-default-{count}-{sampling}', pai.calculate_predict_aspect_importance(
                exp, observation, groups, 20, count, sampling, 2, 42))
    save('aspect-one-group', pai.calculate_predict_aspect_importance(
        exp, observation, {'all': np.array(data.columns)}, 20, None, 'default', 2, 42))
    for processes in [1, 2]:
        np.random.seed(42)
        save(f'aspect-shap-{processes}', pai.calculate_shap_predict_aspect_importance(
            exp, observation, groups, 20, 3, processes, 42))
    save('aspect-path', pai.get_single_random_path(exp.predict, data.iloc[:20], observation,
                                                groups, np.array([1, 0]), 7))
    aspect = dx.Aspect(exp)
    save('model-hierarchy', mt.calculate_model_hierarchical_importance(
        aspect, 'rmse', 'variable_importance', 20, 3, 1, 42))
    save('model-single', mt.calculate_single_variable_importance(
        aspect, 'rmse', 'variable_importance', 20, 3, 1, 42))
    for kind in ['default', 'shap']:
        np.random.seed(42)
        save(f'predict-hierarchy-{kind}', pt.calculate_predict_hierarchical_importance(
            aspect, observation, kind, 20, 3, 'default', 2, 1, 42))
        np.random.seed(42)
        save(f'predict-single-{kind}', pt.calculate_single_variable_importance(
            aspect, observation, kind, 20, 3, 'default', 2, 1, 42))
        indexed = dx.Explainer(model, data, y, predict_function=indexed_predict,
                                model_type='regression', verbose=False, label='indexed')
        save(f'indexed-public-aspect-{kind}', dx.Aspect(indexed).predict_parts(
            observation, variable_groups=groups, type=kind, N=20, B=3, random_state=42).result)
    from dalex.fairness import GroupFairnessClassification, GroupFairnessRegression
    protected = np.tile(['c', 'a', 'b', 'a'], 8)
    for regression in [False, True]:
        for edit in ['none', 'nan', 'nullable', 'duplicate']:
            value = (GroupFairnessRegression(y, exp.y_hat, protected, 'a', 'model', verbose=False) if regression else
                     GroupFairnessClassification((y > np.median(y)).astype(int),
                         1 / (1 + np.exp(-exp.y_hat)), protected, 'a', 'model', verbose=False))
            metric = 'independence' if regression else 'TPR'
            if edit != 'none':
                value.result.index.name = 'edited-group'
                if edit == 'nan':
                    value.result.loc['c', metric] = np.nan
                elif edit == 'nullable':
                    value.result[metric] = value.result[metric].astype('Float64')
                    value.result.loc['c', metric] = pd.NA
                else:
                    value.result = pd.concat([value.result.iloc[::-1], value.result.loc[['c']]])
            buffer = StringIO()
            with redirect_stdout(buffer):
                value.fairness_check(epsilon=.85, verbose=True)
            save(f'fairness-console-{regression}-{edit}', buffer.getvalue())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root')
    parser.add_argument('--reference-root')
    parser.add_argument('--output', required=True)
    parser.add_argument('--labels-only', action='store_true',
                        help='Isolate label contracts from unrelated legacy dependency failures')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.capture_root:
        output.write_bytes(pickle.dumps(capture(args.capture_root, labels_only=args.labels_only)))
        return
    roots = [('reference', Path(args.reference_root).resolve()), ('current', Path(__file__).resolve().parents[1])]
    captures = []
    for name, root in roots:
        path = output.with_suffix(f'.{name}.pkl')
        command = [sys.executable, __file__, '--capture-root', str(root), '--output', str(path)]
        if args.labels_only:
            command.append('--labels-only')
        subprocess.run(command, check=True,
                       env=dict(os.environ, PYTHONHASHSEED='0'))
        captures.append(pickle.loads(path.read_bytes()))
    import pandas as pd
    import numpy as np
    import dalex as dx
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
    from _benchmark_utils import source_identity

    def compare(a, b):
        if isinstance(a, pd.DataFrame):
            pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)
        elif isinstance(a, pd.Series):
            pd.testing.assert_series_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                compare(a[key], b[key])
        elif a is None or isinstance(a, str):
            assert a == b
        elif np.asarray(a).dtype.kind in 'biufc':
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        else:
            np.testing.assert_array_equal(a, b)

    reference, current = captures
    assert reference.keys() == current.keys()
    for name in reference:
        try:
            compare(reference[name], current[name])
        except (AssertionError, TypeError) as error:
            raise AssertionError(f'Legacy adapter mismatch: {name}') from error
    report = {'passed': True, 'outputs': list(reference), 'output_count': len(reference),
              'current_source': source_identity(dx), 'tolerance': {'rtol': 1e-12, 'atol': 1e-12},
              'scope': ('pandas feature/group labels, values, indexes, dtypes and metadata' if args.labels_only else
                        'private helper and pandas label contracts, indexes, dtypes, metadata; index-sensitive public Aspect callbacks'),
              'reference_runtime_shim': 'removed numpy.in1d spelling uses numpy.isin on 1-D feature lists'}
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
