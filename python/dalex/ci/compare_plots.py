"""Compare complete figure payloads with a separate, unmodified pandas source.

Includes model fits, multiple models, limits/baselines, tie ordering, sampled
residuals and classification curves. No screenshot comparison is used. The
unmodified reference's dendrogram/triplot path requires working Kaleido/browser
layout resolution; the migrated implementation no longer requires that path.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def canonical_triplot(figure, *, first_trace=1):
    """Remove only redundant, unlabelled points on monotone straight segments.

    Upstream asks a browser to resolve autorange padding before adding hover
    samples. DALEX now uses explicit finite spans. Segment geometry, labelled
    points and every other figure field must still match independently.
    """
    import base64
    from copy import deepcopy
    import numpy as np
    figure = deepcopy(figure)

    def array(value):
        if isinstance(value, dict) and 'bdata' in value:
            return np.frombuffer(base64.b64decode(value['bdata']), dtype=value['dtype'])
        return np.asarray(value)

    for trace in figure['data'][first_trace:]:
        x, y = array(trace['x']), array(trace['y'])
        text = trace.get('text', [None] * len(x))
        assert len(x) == len(y) == len(text)
        keep = [0]
        for index in range(1, len(x) - 1):
            before, after = index - 1, index + 1
            vertical = x[before] == x[index] == x[after]
            horizontal = y[before] == y[index] == y[after]
            monotone = ((x[index] - x[before]) * (x[after] - x[index]) >= 0 and
                        (y[index] - y[before]) * (y[after] - y[index]) >= 0)
            if text[index] is not None or not (monotone and (vertical or horizontal)):
                keep.append(index)
        keep.append(len(x) - 1)
        trace['x'], trace['y'] = x[keep].tolist(), y[keep].tolist()
        if 'text' in trace:
            trace['text'] = [text[index] for index in keep]
    return figure


def capture(root):
    sys.path.insert(0, str(Path(root).resolve()))
    import numpy as np
    import pandas as pd
    import dalex as dx
    from copy import deepcopy
    from dalex.fairness import GroupFairnessClassification, GroupFairnessRegression
    from sklearn.linear_model import LinearRegression
    # Untouched upstream still uses the removed spelling on 1-D feature lists.
    # This runtime shim changes no dataframe or explanation implementation.
    if not hasattr(np, 'in1d'):
        def in1d(array, values, *args, **kwargs):
            assert np.asarray(array).ndim == 1
            return np.isin(array, values, *args, **kwargs)
        np.in1d = in1d

    rng = np.random.default_rng(391)
    data = pd.DataFrame(rng.normal(size=(40, 3)), columns=['z', 'a', 'b'], index=np.arange(40) * 3 + 7)
    y = data.z * 2 + data.a / 3 + np.sin(data.b)
    exps = [dx.Explainer(LinearRegression().fit(data, y + offset * data.a), data, y,
                         label=label, verbose=False) for label, offset in [('second', 0), ('first', .2)]]
    figures = {}

    def save(name, figure):
        figures[name] = json.loads(figure.to_json())

    for include in [False, True]:
        for count in [1, 2]:
            for multiple in [False, True]:
                cp = [exp.predict_profile(data.iloc[[3, 7][:count]], variables=['z', 'a'] if multiple else ['z'],
                    grid_points=5, variable_splits_with_obs=include, verbose=False) for exp in exps]
                for color in ['_label_', '_ids_']:
                    for points in [False, True]:
                        save(f'cp-{include}-{count}-{multiple}-{color}-{points}', cp[0].plot(cp[1],
                            color=color, show_observations=points, show=False))
    nullable_data = data.round().astype('Float64')
    nullable = dx.Explainer(exps[0].model, nullable_data, y, label='nullable', verbose=False)
    save('cp-nullable-floating-captions', nullable.predict_profile(nullable_data.iloc[[3]],
        variables=['z', 'a'], grid_points=3, verbose=False).plot(show=False))

    for family in ['partial', 'conditional', 'accumulated']:
        profiles = [exp.model_profile(type=family, N=None, variables=['z', 'a'], grid_points=5,
                                       random_state=42, verbose=False) for exp in exps]
        for geom in ['aggregates', 'bars', 'profiles']:
            for center in [False, True]:
                save(f'profiles-{family}-{geom}-{center}', profiles[0].plot(
                    profiles[1], geom=geom, center=center, variables=['a', 'z'], show=False))
    grouped_data = data.assign(group=np.where(data.z > 0, 'positive', 'negative'))
    grouped = dx.Explainer(exps[0].model, grouped_data, y, label='grouped', verbose=False,
                            predict_function=lambda model, batch: model.predict(batch[['z', 'a', 'b']]))
    for count in [1, 2]:
        for multiple in [False, True]:
            cp = grouped.predict_profile(grouped_data.iloc[[3, 7][:count]],
                variables=['group', 'z'] if multiple else ['group'], grid_points=5,
                variable_splits_with_obs=True, verbose=False)
            for color in ['_label_', '_ids_']:
                for points in [False, True]:
                    save(f'cp-categorical-{count}-{multiple}-{color}-{points}', cp.plot(variable_type='categorical',
                        variables=['group', 'z'] if multiple else None, color=color, show_observations=points, show=False))
    for family in ['partial', 'conditional', 'accumulated']:
        for categorical in [False, True]:
            profile = grouped.model_profile(type=family, N=None,
                variables=['group'] if categorical else ['z'],
                variable_type='categorical' if categorical else 'numerical',
                groups=None if categorical else ['group'], grid_points=5, random_state=42, verbose=False)
            for center in [False, True]:
                save(f'grouped-profiles-{family}-{categorical}-{center}', profile.plot(center=center, show=False))

    importance = [exp.model_parts(N=30, B=3, random_state=42) for exp in exps]
    for split in ['model', 'variable']:
        for limit in [1, 2, None]:
            save(f'importance-{split}-{limit}', importance[0].plot(importance[1], split=split, max_vars=limit, show=False))
    for mixed in [False, True]:
        # More than sixteen equal importances exercises pandas' quicksort tie
        # behavior; mixed numeric/string group labels are a compatibility case.
        names = list(range(10)) + [f'x{i}' for i in range(10)] if mixed else [f'x{i}' for i in range(20)]
        parts = []
        for label in ['model1', 'model2']:
            value = dx.model_explanations.VariableImportance()
            value.result = pd.DataFrame({'variable': ['_baseline_', '_full_model_', *names],
                                         'dropout_loss': [3., .1, *([1.] * 20)], 'label': label})
            parts.append(value)
        # Plotly subplot titles require strings; the model split still preserves
        # arbitrary variable label values on the axis.
        for limit in [5, None]:
            save(f'importance-ties-{mixed}-{limit}', parts[0].plot(parts[1], max_vars=limit, show=False))
    diagnostics = [exp.model_diagnostics() for exp in exps]
    for smooth in [True, False]:
        for count in [10, None]:
            save(f'diagnostics-{smooth}-{count}', diagnostics[0].plot(diagnostics[1], smooth=smooth, N=count, show=False))
    for family in ['break_down', 'break_down_interactions', 'shap']:
        parts = [exp.predict_parts(data.iloc[[3]], type=family, N=30, B=3, random_state=42) for exp in exps]
        for limit in [1, 2, 20]:
            for baseline in [None, 0, 1.5]:
                save(f'{family}-{limit}-{baseline}', parts[0].plot(parts[1], max_vars=limit, baseline=baseline, show=False))
    for count in [103, 501]:
        values = rng.uniform(size=count)
        for ties in [True, False]:
            parts = []
            for label in ['second', 'first']:
                target = (values > .4).astype(int)
                prediction = np.round(values, 1) if ties else values
                value = dx.model_explanations.ModelPerformance('classification')
                value.residuals = pd.DataFrame({'y': target, 'y_hat': prediction,
                                               'residuals': target - prediction, 'label': label})
                parts.append(value)
            for geom in ['roc', 'ecdf', 'lift']:
                np.random.seed(938)
                save(f'performance-{count}-{ties}-{geom}', parts[0].plot(parts[1], geom=geom, show=False))
    protected = np.repeat(['a', 'b', 'c'], 40)
    target = np.tile(np.arange(40) % 2, 3)
    probabilities = np.tile(np.linspace(.05, .95, 40), 3)
    fairness = [GroupFairnessClassification(target, probabilities + offset, protected, 'a', label)
                for label, offset in [('second', 0), ('first', .025)]]
    for multiple in [False, True]:
        for kind in ['fairness_check', 'metric_scores', 'stacked', 'radar', 'performance_and_fairness',
                     'heatmap', 'density', 'ceteris_paribus_cutoff']:
            pair = deepcopy(fairness)
            options = {} if kind == 'density' else {'verbose': False}
            if kind == 'ceteris_paribus_cutoff':
                options.update(subgroup='b', grid_points=7, metrics=['TPR', 'ACC'])
            save(f'fairness-{multiple}-{kind}', pair[0].plot(pair[1] if multiple else None,
                                                            type=kind, show=False, **options))
    fair_rng = np.random.default_rng(842)
    target = fair_rng.normal(size=120)
    predictions = .8 * target + fair_rng.normal(size=120) * .4
    fairness = [GroupFairnessRegression(target, predictions + offset, protected, 'a', label)
                for label, offset in [('second', 0), ('first', .025)]]
    for multiple in [False, True]:
        for kind in ['fairness_check', 'density']:
            options = {} if kind == 'density' else {'verbose': False}
            save(f'regression-fairness-{multiple}-{kind}', fairness[0].plot(fairness[1] if multiple else None,
                                                                          type=kind, show=False, **options))
    aspects = [dx.Aspect(exp) for exp in exps]
    for digits in [0, 3]:
        save(f'aspect-dendrogram-{digits}', aspects[0].plot_dendrogram(digits=digits, show=False))
    save('aspect-dendrogram-custom', aspects[0].plot_dendrogram(
        title='Custom dendrogram', lines_interspace=37, digits=2,
        rounding_function=lambda x, digits: np.ceil(x * 10**digits) / 10**digits, show=False))
    constant_data = pd.DataFrame(np.zeros((8, 3)), columns=['z', 'a', 'b'])
    constant = dx.Aspect(dx.Explainer(LinearRegression().fit(constant_data, np.zeros(8)),
        constant_data, np.zeros(8), label='constant', verbose=False))
    save('aspect-dendrogram-constant', constant.plot_dendrogram(show=False))
    groups = {'za': ['z', 'a'], 'b': ['b']}
    model_parts = [aspect.model_parts(variable_groups=groups, N=30, B=3, random_state=42) for aspect in aspects]
    for split in ['model', 'aspect']:
        for limit in [1, None]:
            save(f'aspect-model-{split}-{limit}', model_parts[0].plot(model_parts[1], split=split,
                                                                     max_aspects=limit, show=False))
    for kind in ['default', 'shap']:
        parts = [aspect.predict_parts(data.iloc[[3]], variable_groups=groups, type=kind,
                                       N=30, B=3, random_state=42) for aspect in aspects]
        for limit in [1, 10]:
            for baseline in [None, 0., 1.5]:
                save(f'aspect-predict-{kind}-{limit}-{baseline}', parts[0].plot(parts[1],
                    max_aspects=limit, baseline=baseline, show=False))
        triplot = aspects[0].predict_triplot(data.iloc[[3]], type=kind, N=30, B=3, random_state=42)
        for absolute in [False, True]:
            save(f'aspect-predict-triplot-{kind}-{absolute}', triplot.plot(
                absolute_value=absolute, widget=False, show=False))
    triplot = aspects[0].model_triplot(N=30, B=3, random_state=42)
    for change in [False, True]:
        save(f'aspect-model-triplot-{change}', triplot.plot(show_change=change, widget=False, show=False))
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root')
    parser.add_argument('--capture-root')
    parser.add_argument('--output', default='plot-parity.json')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.capture_root:
        output.write_text(json.dumps(capture(args.capture_root)), encoding='utf-8')
        return
    roots = [('reference', Path(args.reference_root).resolve()), ('current', Path(__file__).resolve().parents[1])]
    captures = []
    for name, root in roots:
        path = output.with_suffix(f'.{name}.json')
        subprocess.run([sys.executable, __file__, '--capture-root', str(root), '--output', str(path)], check=True,
                       env=dict(os.environ, PYTHONHASHSEED='0'))
        captures.append(json.loads(path.read_text(encoding='utf-8')))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
    from bench_plot_preparation import compare_figures
    from _benchmark_utils import source_identity
    import dalex as dx
    a, b = captures
    assert a.keys() == b.keys()
    for name in a:
        try:
            if name.startswith(('aspect-predict-triplot-', 'aspect-model-triplot-')):
                compare_figures(canonical_triplot(a[name]), canonical_triplot(b[name]))
            elif name.startswith('aspect-dendrogram-'):
                compare_figures(canonical_triplot(a[name], first_trace=0),
                                canonical_triplot(b[name], first_trace=0))
            else:
                compare_figures(a[name], b[name])
        except AssertionError as error:
            raise AssertionError(f'Figure mismatch: {name}') from error
    report = {'passed': True, 'figure_count': len(a), 'figures': list(a),
              'reference_root': str(roots[0][1]), 'current_source': source_identity(dx),
              'tolerance': {'rtol': 1e-12, 'atol': 1e-12},
              'scope': 'all figure fields; six triplots and four base dendrograms normalize redundant unlabelled collinear hover samples',
              'triplot_hover_density_exception': [name for name in a if name.startswith(
                  ('aspect-predict-triplot-', 'aspect-model-triplot-'))],
              'dendrogram_hover_density_exception': [name for name in a if name.startswith('aspect-dendrogram-')],
              'reference_runtime_shim': 'numpy.in1d uses numpy.isin for one-dimensional feature lists when absent'}
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
