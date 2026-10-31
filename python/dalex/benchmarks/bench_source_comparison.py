"""Alternate immutable DALEX sources using the same fitted model and data.

Each trial is an isolated interpreter with an excluded warm-up. Model fitting,
original Polars creation and process startup are excluded. The timed workflow
includes original-frame conversion, Explainer initialization, explanation fitting
and compatible pandas result publication. This is not a free-threading benchmark.
"""
import argparse
import gc
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from time import perf_counter


def dataset(rows, features):
    import numpy as np
    import polars as pl
    rng = np.random.default_rng(123)
    values = rng.normal(size=(rows, features))
    target = values[:, :5].sum(axis=1) + rng.normal(size=rows) * .1
    return values, target, pl.DataFrame(values, schema=[f'x{i}' for i in range(features)])


def explain(exp, family, sample, rounds):
    if family == 'importance':
        return exp.model_parts(N=sample, B=rounds, random_state=42).result
    if family in ['breakdown', 'shap']:
        return exp.predict_parts(exp.data.head(1), type='break_down' if family == 'breakdown' else 'shap',
                                  N=sample, B=rounds, random_state=42).result
    return exp.model_profile(type=family, N=sample, variables=['x0', 'x1'],
                              grid_points=10, random_state=42, verbose=False).result


def worker(args):
    sys.path.insert(0, str(Path(args.worker_root).resolve()))
    import dalex as dx
    import numpy as np
    import polars as pl
    from threadpoolctl import threadpool_limits
    from _benchmark_utils import RSS, source_identity
    values, target, original = dataset(args.rows, args.features)
    model_bytes = Path(args.model_file).read_bytes()
    model = pickle.loads(model_bytes)
    assert model.n_features_in_ == args.features
    assert type(model).__module__.startswith(args.model)
    with threadpool_limits(limits=1):
        for warm in [True, False]:
            gc.collect()
            with RSS() as rss:
                start = perf_counter()
                data = original.to_pandas() if args.backend == 'pandas' else original.clone()
                converted = perf_counter()
                exp = dx.Explainer(model, data, target, model_type='regression', label='model',
                                   precalculate=args.precalculate, verbose=False)
                initialized = perf_counter()
                result = explain(exp, args.family, min(args.rows, args.sample), args.rounds)
                finished = perf_counter()
            if warm:
                del data, exp, result
    numeric = result.select_dtypes(include=np.number)
    return {'backend': args.backend, 'family': args.family, 'seconds': finished - start,
            'conversion_seconds': converted - start, 'initialization_seconds': initialized - converted,
            'explanation_seconds': finished - initialized, 'rss_baseline_mib': rss.baseline / 2**20,
            'rss_peak_mib': rss.peak / 2**20, 'rss_increase_mib': (rss.peak - rss.baseline) / 2**20,
            'outcome': {'columns': list(numeric.columns), 'values': numeric.to_numpy(dtype=float).tolist(),
                        'metadata': json.loads(result.select_dtypes(exclude=np.number).to_json(
                            orient='split', date_format='iso', double_precision=15))},
            'source': source_identity(dx), 'fitted_model_sha256': hashlib.sha256(model_bytes).hexdigest(),
            'data_sha256': hashlib.sha256(values.tobytes() + target.tobytes()).hexdigest(),
            'polars_thread_pool_size': pl.thread_pool_size(), 'model_and_blas_threads': 1,
            'gil_enabled': getattr(sys, '_is_gil_enabled', lambda: None)(), 'python': sys.version,
            'versions': {name: metadata.version(name) for name in
                ['pandas', 'polars', 'pyarrow', 'narwhals', 'numpy', 'scikit-learn'] +
                (['lightgbm'] if args.model == 'lightgbm' else [])}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', default=[], help='NAME=Python package root')
    parser.add_argument('--revision', action='append', default=[], help='NAME=Git revision')
    parser.add_argument('--pandas-only-source', action='append', default=[])
    parser.add_argument('--model', choices=['sklearn', 'lightgbm'], required=True)
    parser.add_argument('--model-file', required=True)
    parser.add_argument('--rows', type=int, default=1000000)
    parser.add_argument('--features', type=int, default=10)
    parser.add_argument('--sample', type=int, default=1000)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--precalculate', action=argparse.BooleanOptionalAction, default=False,
                        help='Include normal full-reference prediction/residual precalculation in initialization')
    parser.add_argument('--families', nargs='+', default=['importance', 'breakdown', 'shap', 'partial', 'conditional', 'accumulated'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--worker-root')
    parser.add_argument('--backend')
    parser.add_argument('--family')
    args = parser.parse_args()
    assert args.rows > 0 and args.features >= 2 and args.sample > 0 and args.rounds > 0 and args.trials > 0
    output = Path(args.output).resolve()
    if args.worker_root:
        output.write_text(json.dumps(worker(args), indent=2), encoding='utf-8')
        return
    import numpy as np
    from threadpoolctl import threadpool_limits
    from _benchmark_utils import summary
    sources = dict(value.split('=', 1) for value in args.source)
    revisions = dict(value.split('=', 1) for value in args.revision)
    assert sources and set(args.pandas_only_source) <= sources.keys()
    assert set(sources) == revisions.keys(), 'Name every measured source revision explicitly'
    model_file = Path(args.model_file).resolve()
    values, target, original = dataset(args.rows, args.features)
    data_hash = hashlib.sha256(values.tobytes() + target.tobytes()).hexdigest()
    if not model_file.exists():
        with threadpool_limits(limits=1):
            if args.model == 'sklearn':
                from sklearn.ensemble import HistGradientBoostingRegressor
                model = HistGradientBoostingRegressor(max_iter=20, random_state=42)
            else:
                from lightgbm import LGBMRegressor
                model = LGBMRegressor(n_estimators=20, n_jobs=1, random_state=42, verbosity=-1)
            model.fit(original, target)
        model_file.write_bytes(pickle.dumps(model))
        model_file.with_suffix('.data-sha256').write_text(data_hash, encoding='utf-8')
    assert model_file.with_suffix('.data-sha256').read_text(encoding='utf-8') == data_hash, 'Cached model data mismatch'
    model_hash = hashlib.sha256(model_file.read_bytes()).hexdigest()
    del values, target, original
    gc.collect()
    report = {'protocol': __doc__, 'settings': vars(args), 'sources': {}, 'trials': [],
              'outcomes': {}, 'summaries': [], 'max_absolute_parity_error': {},
              'fitted_model_sha256': model_hash, 'data_sha256': data_hash,
              'rss_sampling_seconds': .002, 'source_order': 'alternating forward/reverse per trial'}
    temporary = output.with_suffix('.worker.json')
    for family in args.families:
        expected = None
        error = 0.
        for trial in range(args.trials):
            names = list(sources) if trial % 2 == 0 else list(reversed(sources))
            for source in names:
                backends = ['pandas'] if source in args.pandas_only_source else ['pandas', 'polars']
                if trial % 2:
                    backends.reverse()
                for backend in backends:
                    command = [sys.executable, __file__, '--worker-root', sources[source], '--backend', backend,
                        '--family', family, '--model', args.model, '--model-file', str(model_file),
                        '--rows', str(args.rows), '--features', str(args.features), '--sample', str(args.sample),
                        '--rounds', str(args.rounds), '--output', str(temporary)]
                    command.append('--precalculate' if args.precalculate else '--no-precalculate')
                    subprocess.run(command, check=True, env=dict(os.environ, PYTHONHASHSEED='0'),
                                   stdout=subprocess.DEVNULL)
                    item = json.loads(temporary.read_text(encoding='utf-8'))
                    assert item['data_sha256'] == data_hash and item['fitted_model_sha256'] == model_hash
                    identity = {'revision': revisions[source], **item['source']}
                    if source in report['sources']:
                        assert report['sources'][source] == identity, 'Source changed during measurements'
                    report['sources'][source] = identity
                    outcome = item.pop('outcome')
                    if expected is None:
                        expected = outcome
                    assert outcome['columns'] == expected['columns']
                    assert outcome['metadata'] == expected['metadata'], 'Result labels/index differ across sources or backends'
                    left, right = np.asarray(outcome['values']), np.asarray(expected['values'])
                    np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-10, equal_nan=True)
                    error = max(error, float(np.nanmax(np.abs(left - right))))
                    report['outcomes'][f'{family}/{source}/{backend}'] = outcome
                    report['trials'].append(dict(item, source_name=source, trial=trial))
                    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(f'{args.model}/{family}: trial {trial + 1}/{args.trials}, max numerical error {error}', flush=True)
        report['max_absolute_parity_error'][family] = error
    measures = ['seconds', 'conversion_seconds', 'initialization_seconds', 'explanation_seconds',
                'rss_baseline_mib', 'rss_peak_mib', 'rss_increase_mib']
    for family in args.families:
        for source in sources:
            for backend in ['pandas', 'polars']:
                selected = [item for item in report['trials'] if item['family'] == family and
                            item['source_name'] == source and item['backend'] == backend]
                if selected:
                    report['summaries'].append({'family': family, 'source': source, 'backend': backend,
                        **{measure: summary([item[measure] for item in selected]) for measure in measures}})
    report['passed'] = True
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
