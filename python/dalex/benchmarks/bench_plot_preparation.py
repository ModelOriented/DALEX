"""Isolated plotting-only measurements of identical precomputed explanations.

Compare a reference source tree with this checkout. This does not measure model
training or explanation fitting, and cannot establish end-to-end model speedups.
Each trial has a separate interpreter, an excluded warm-up, and a fresh result
whose lazy compatibility table has not been read. RSS is sampled every 2 ms.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(args):
    sys.path.insert(0, str(Path(args.root).resolve()))
    import dalex as dx
    import narwhals
    import narwhals.stable.v2 as nw
    import numpy as np
    import pandas as pd
    import polars as pl
    import pyarrow as pa
    from dalex._explanation import IndexedTable
    from dalex.predict_explanations._native_contributions import ShapTable
    from _benchmark_utils import RSS, source_identity

    rng = np.random.default_rng(391)
    backend = args.backend

    def native(data):
        return nw.from_dict(data, backend=backend).to_native()

    if args.family == 'importance':
        columns = {'variable': ['_full_model_', '_baseline_', *[f'x{i}' for i in range(500)]],
                   'dropout_loss': np.r_[.1, 3., rng.uniform(.1, 2., 500)], 'label': ['model'] * 502}

        def make():
            value = dx.model_explanations.VariableImportance()
            value.result = native(columns)
            return value

    elif args.family == 'diagnostics':
        rows = 100000
        columns = {f'x{i}': rng.normal(size=rows) for i in range(10)}
        columns.update(y_hat=rng.normal(size=rows), residuals=rng.normal(size=rows),
                       ids=np.arange(rows) + 1, label=['model'] * rows)

        def make():
            value = dx.model_explanations.ResidualDiagnostics()
            value.result = IndexedTable(native(columns))
            return value

    elif args.family == 'performance':
        rows = 100000
        predicted = rng.uniform(size=rows)
        target = (predicted > .4).astype(int)
        columns = {'y_hat': predicted, 'y': target, 'residuals': target - predicted, 'label': ['model'] * rows}

        def make():
            value = dx.model_explanations.ModelPerformance('classification')
            value.residuals = native(columns)
            return value

    elif args.family == 'fairness-check':
        from dalex.fairness import GroupFairnessClassification
        groups, count = 4, 256
        protected = np.repeat([f'g{i}' for i in range(groups)], count)
        target = np.tile(np.arange(count) % 2, groups)
        predicted = np.tile(np.linspace(.05, .95, count), groups) + np.repeat([0., .01, -.01, .02], count)

        def make():
            first = GroupFairnessClassification(target, predicted, protected, 'g0', 'model1', _backend=backend)
            other = GroupFairnessClassification(target, predicted + .01, protected, 'g0', 'model2', _backend=backend)
            first._benchmark_others = [other]
            return first

    elif args.family in ['ceteris-paribus', 'profiles-overlay']:
        from dalex.predict_explanations._ceteris_paribus._table import ProfileTables, ObservationTable
        from dalex.model_explanations._aggregated_profiles._table import AggregatedPartitions
        observations, features, grid_count = 20, 8, 31
        names = [f'x{i}' for i in range(features)]
        original = rng.normal(size=(observations, features))
        grid = np.linspace(-2., 2., grid_count)
        positions = np.repeat(np.arange(observations), grid_count)
        indexes = [positions.copy() for _ in names]
        cp_frames, ap_frames = [], []
        for index, name in enumerate(names):
            values = original[positions].copy()
            values[:, index] = np.tile(grid, observations)
            predictions = values.sum(axis=1)
            cp_frames.append(native(dict({column: values[:, i] for i, column in enumerate(names)},
                _original_=original[positions, index], _yhat_=predictions, _vname_=[name] * len(positions),
                _ids_=positions, _label_=['model'] * len(positions))))
            ap_frames.append(native({'_vname_': [name] * grid_count, '_label_': ['model'] * grid_count,
                '_x_': grid, '_yhat_': predictions.reshape(observations, grid_count).mean(axis=0)}))
        observation_frame = native(dict({name: original[:, i] for i, name in enumerate(names)},
            _yhat_=original.sum(axis=1), _label_=['model'] * observations, _ids_=np.arange(observations)))

        def make():
            value = dx.predict_explanations.CeterisParibus(variables=names, variable_splits_with_obs=False)
            value.result = ProfileTables(cp_frames, indexes, names)
            value.new_observation = ObservationTable(observation_frame, np.arange(observations))
            if args.family == 'ceteris-paribus':
                return value
            aggregate = dx.model_explanations.AggregatedProfiles()
            aggregate.result = AggregatedPartitions(ap_frames, None, [], {}, True)
            aggregate.mean_prediction = float(original.sum(axis=1).mean())
            aggregate.raw_profiles = value
            return aggregate

    else:
        features, paths = 500, 10
        names = [f'x{i}' for i in range(features)]
        contributions = rng.normal(size=features)
        columns = {'variable': names * paths, 'variable_name': names * paths,
                   'contribution': np.tile(contributions, paths), 'sign': np.tile(np.sign(contributions), paths),
                   'B': np.repeat(np.arange(paths), features), 'label': ['model'] * (features * paths)}
        observation = {name: [float(i)] for i, name in enumerate(names)}

        def make():
            value = dx.predict_explanations.Shap()
            value.result = ShapTable(native(columns), native(observation))
            value.intercept, value.prediction = 0., float(contributions.sum())
            return value

    def plot(value):
        np.random.seed(738)
        if args.family == 'diagnostics':
            return value.plot(N=1000, show=False)
        if args.family == 'performance':
            return value.plot(geom='roc', show=False)
        if args.family == 'ceteris-paribus':
            return value.plot(show_observations=False, show=False)
        if args.family == 'profiles-overlay':
            return value.plot(geom='profiles', show=False)
        if args.family == 'fairness-check':
            return value.plot(value._benchmark_others, type='fairness_check', verbose=False, show=False)
        return value.plot(show=False)

    plot(make())
    value = make()
    gc.collect()
    with RSS() as memory:
        start = time.perf_counter()
        figure = plot(value)
        elapsed = time.perf_counter() - start
    payload = figure.to_json()
    if args.figure:
        Path(args.figure).write_text(payload, encoding='utf-8')
    compatibility_tables = [key for key in value.__dict__ if key.startswith('_compatibility_')]
    raw_profiles = getattr(value, 'raw_profiles', None)
    if raw_profiles is not None:
        compatibility_tables.extend('raw_profiles.' + key for key in raw_profiles.__dict__
                                    if key.startswith('_compatibility_'))
    return {'family': args.family, 'backend': backend, 'elapsed_seconds': elapsed,
            'rss_baseline_mib': memory.baseline / 2**20, 'rss_peak_mib': memory.peak / 2**20,
            'rss_increase_mib': (memory.peak - memory.baseline) / 2**20,
            'figure_sha256': hashlib.sha256(payload.encode()).hexdigest(),
            'materialized_compatibility_tables': sorted(compatibility_tables),
            'source': source_identity(dx), 'python': sys.version,
            'versions': {'pandas': pd.__version__, 'polars': pl.__version__, 'pyarrow': pa.__version__, 'narwhals': narwhals.__version__}}


def compare_figures(left, right):
    """Exact presentation comparison with tolerant floating point coordinates."""
    import base64
    import numpy as np
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        if 'bdata' in left:
            a = np.frombuffer(base64.b64decode(left['bdata']), dtype=left['dtype'])
            b = np.frombuffer(base64.b64decode(right['bdata']), dtype=right['dtype'])
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        else:
            for key in left:
                compare_figures(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            compare_figures(a, b)
    elif isinstance(left, float):
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)
    else:
        assert left == right, (left, right)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root')
    parser.add_argument('--reference-revision', default='unspecified')
    parser.add_argument('--current-revision', default='unspecified')
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--output', required=True)
    parser.add_argument('--root')
    parser.add_argument('--family')
    parser.add_argument('--families', nargs='+', default=['importance', 'diagnostics', 'performance', 'shap'])
    parser.add_argument('--backend')
    parser.add_argument('--figure')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.root:
        output.write_text(json.dumps(worker(args), indent=2), encoding='utf-8')
        return
    from _benchmark_utils import summary
    roots = {'reference': str(Path(args.reference_root).resolve()), 'current': str(Path(__file__).resolve().parents[1])}
    trial_dir = output.parent / (output.stem + '-trials')
    trial_dir.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'plotting only, identical precomputed results; no fitting or training',
              'reference_revision': args.reference_revision, 'current_revision': args.current_revision,
              'threads': {'polars': os.environ.get('POLARS_MAX_THREADS'), 'blas': os.environ.get('OPENBLAS_NUM_THREADS')},
              'trials': [], 'summaries': [], 'figure_parity': []}
    for family in args.families:
        for backend in ['pandas', 'polars', 'pyarrow']:
            for trial in range(args.trials):
                for version in (['reference', 'current'] if trial % 2 == 0 else ['current', 'reference']):
                    result = trial_dir / f'{family}-{backend}-{trial}-{version}.json'
                    figure = result.with_suffix('.figure.json')
                    subprocess.run([sys.executable, __file__, '--root', roots[version], '--family', family,
                                    '--backend', backend, '--output', str(result), '--figure', str(figure)], check=True)
                    measured = json.loads(result.read_text(encoding='utf-8'))
                    report['trials'].append(dict(measured, trial=trial, version=version))
                a, b = [json.loads((trial_dir / f'{family}-{backend}-{trial}-{version}.figure.json').read_text(encoding='utf-8'))
                        for version in ['reference', 'current']]
                compare_figures(a, b)
            for version in roots:
                samples = [value for value in report['trials'] if value['family'] == family and value['backend'] == backend and value['version'] == version]
                report['summaries'].append(dict(family=family, backend=backend, version=version,
                    **{name: summary([sample[name] for sample in samples]) for name in
                       ['elapsed_seconds', 'rss_baseline_mib', 'rss_peak_mib', 'rss_increase_mib']}))
            report['figure_parity'].append({'family': family, 'backend': backend, 'passed': True, 'trials': args.trials})
            output.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(f'Completed {family}/{backend}', flush=True)


if __name__ == '__main__':
    main()
