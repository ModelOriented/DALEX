"""Reproducible end-to-end and dataframe-only trials; ratios may favor either backend.

Each trial runs in a fresh spawned process. Timed regions include constructing the
native frame from identical NumPy values; model training and process startup are
excluded. RSS is sampled every 2ms (an observational lower bound on true peak).
"""
import argparse
import gc
import importlib.metadata
import json
import multiprocessing as mp
import platform
import sys
import traceback
import pickle
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

import dalex as dx
from dalex._dataframe import shuffle_columns, take_rows
from _benchmark_utils import RSS, summary, source_identity


class TimedPredict:
    def __init__(self, native=False):
        self.seconds = 0.
        self.native = native

    def __call__(self, model, frame):
        start = perf_counter()
        # The identical fitted estimator consumes NumPy on both backends.
        # This explicit model boundary conversion is included in measured time.
        values = frame if self.native or isinstance(frame, np.ndarray) else frame.to_numpy()
        result = model.predict(values)
        self.seconds += perf_counter() - start
        return result


def trial(queue, backend, values, y, weights, model, rounds, warm_runtime=False, native_predict=False, sample_limit=10_000):
    try:
        with threadpool_limits(limits=1):
            names = [f'x{i}' for i in range(values.shape[1])]
            def make_frame():
                return (pd.DataFrame(values.copy(), columns=names) if backend == 'pandas'
                        else pl.DataFrame(values.copy(), schema=names))
            if warm_runtime:
                # Warm this process, not a different discarded worker. The actual
                # timed region still creates a fresh native frame from values.
                warm_frame = make_frame()
                warm_explainer = dx.Explainer(
                    model, warm_frame, y, weights=weights, predict_function=TimedPredict(native_predict),
                    model_type='regression', verbose=False, precalculate=False,
                )
                warm_explainer.model_parts(
                    N=min(len(y), sample_limit), B=rounds, random_state=42,
                    keep_raw_permutations=True, loss_function=weighted_rmse,
                )
                del warm_frame, warm_explainer
            gc.collect()
            predict = TimedPredict(native_predict)
            with RSS() as rss:
                start = perf_counter()
                frame = make_frame()
                explainer = dx.Explainer(model, frame, y, weights=weights, predict_function=predict,
                                         model_type='regression', verbose=False, precalculate=False)
                result = explainer.model_parts(N=min(len(y), sample_limit), B=rounds,
                                               random_state=42, keep_raw_permutations=True,
                                               loss_function=weighted_rmse)
                elapsed = perf_counter() - start
            end_to_end = {'seconds': elapsed, 'inference_seconds': predict.seconds,
                          'peak_rss_bytes': rss.peak, 'rss_increase_bytes': max(0, rss.peak-rss.baseline)}
            # Independently measure adapter work including native construction.
            del frame, explainer
            gc.collect()
            rng = np.random.default_rng(42)
            positions = rng.choice(len(y), min(len(y), sample_limit), replace=False)
            permutations = [rng.permutation(len(positions)) for _ in range(rounds)]
            with RSS() as rss:
                start = perf_counter()
                frame = take_rows(make_frame(), positions)
                for order in permutations:
                    for name in names:
                        shuffled = shuffle_columns(frame, name, order)
                dataframe_seconds = perf_counter()-start
            raw = result.permutation
            if raw is None:
                # DALEX omits the raw table for B=1; its means are exactly the
                # one observed permutation, so parity remains checkable.
                raw = result.result.set_index('variable')[['dropout_loss']].T
            queue.put({'backend': backend, 'end_to_end': end_to_end,
                       'dataframe': {'seconds': dataframe_seconds, 'peak_rss_bytes': rss.peak,
                                     'rss_increase_bytes': max(0, rss.peak-rss.baseline)},
                       'outcome': result.result.sort_values('variable').dropout_loss.tolist(),
                       'raw': raw.sort_index(axis=1).to_numpy().tolist()})
    except BaseException:
        queue.put({'error': traceback.format_exc()})


def weighted_rmse(y, prediction, sample_weight=None):
    return np.sqrt(np.average((y-prediction)**2, weights=sample_weight))


def run(sizes, n_features, repetitions, rounds, warm_runtime=False, output_prefix="dalex-narwhals-benchmark", model_kind="sklearn-numpy", sample_limit=10_000, source_revision="unspecified"):
    report = {'environment': {'python': sys.version, 'platform': platform.platform(),
              'polars_thread_pool_size': pl.thread_pool_size(),
              'gil_enabled': getattr(sys, '_is_gil_enabled', lambda: None)(),
              'versions': {name: importlib.metadata.version(name) for name in
                           ('dalex','numpy','pandas','polars','narwhals','scipy','scikit-learn','psutil')}},
              'method': {'repetitions': repetitions, 'rounds': rounds, 'features': n_features,
                         'seed_data': 123, 'seed_explanation': 42, 'sample_limit': sample_limit,
                         'threads_estimator': 1, 'rss_poll_seconds': .002,
                         'conversion_included': True, 'warmup_trials_per_backend': 1, 'warm_runtime_in_each_worker': warm_runtime,
                         'memory_note': 'Sampled process RSS, including interpreter/dependencies; peak may be underestimated.'},
              'source_revision': source_revision, 'library_source': source_identity(dx), 'datasets': []}
    report['method']['model_kind'] = model_kind
    report['method']['training_input'] = 'numpy' if model_kind == 'sklearn-numpy' else 'polars'
    if model_kind == 'lightgbm-native':
        report['environment']['versions']['lightgbm'] = importlib.metadata.version('lightgbm')
    context = mp.get_context('spawn')
    for n in sizes:
        rng = np.random.default_rng(123)
        values = rng.normal(size=(n, n_features))
        y = values[:, :min(5,n_features)].sum(axis=1) + rng.normal(size=n)*.1
        weights = np.linspace(1, 2, n)
        with threadpool_limits(limits=1):
            training = values if model_kind == 'sklearn-numpy' else pl.DataFrame(values, schema=[f'x{i}' for i in range(n_features)])
            if model_kind == 'lightgbm-native':
                from lightgbm import LGBMRegressor
                model = LGBMRegressor(n_estimators=20, n_jobs=1, random_state=42, verbosity=-1).fit(training, y)
            else:
                model = HistGradientBoostingRegressor(max_iter=20, random_state=42).fit(training, y)
        trials = []
        expected = None
        max_error = 0.
        for repeat in range(-1, repetitions):
            for backend in (('pandas','polars') if repeat % 2 == 0 else ('polars','pandas')):
                queue = context.Queue()
                worker = context.Process(target=trial, args=(queue,backend,values,y,weights,model,rounds,warm_runtime,model_kind != 'sklearn-numpy',sample_limit))
                worker.start()
                item = queue.get(timeout=600)
                worker.join(timeout=30)
                if worker.exitcode != 0 or 'error' in item:
                    raise RuntimeError(item)
                raw = np.array(item.pop('raw'))
                outcome = np.array(item.pop('outcome'))
                if expected is None:
                    expected = (raw, outcome)
                for actual, reference in zip((raw,outcome),expected):
                    np.testing.assert_allclose(actual, reference, atol=1e-10, rtol=1e-10)
                    max_error = max(max_error, float(np.max(np.abs(actual-reference))))
                if repeat >= 0:
                    item['repetition'] = repeat
                    trials.append(item)
        stats = {}
        for backend in ('pandas','polars'):
            items = [t for t in trials if t['backend']==backend]
            stats[backend] = {phase: {key: summary([t[phase][key] for t in items])
                              for key in items[0][phase]} for phase in ('end_to_end','dataframe')}
        ratios = {phase: stats['pandas'][phase]['seconds']['median']/stats['polars'][phase]['seconds']['median']
                  for phase in ('end_to_end','dataframe')}
        report['datasets'].append({'rows':n,'max_absolute_parity_error':max_error,
                                   'fitted_model_sha256':hashlib.sha256(pickle.dumps(model)).hexdigest(),
                                   'passed_numerical_parity':True,'trials':trials,'summary':stats,
                                   'pandas_time_over_polars_time':ratios})
        print(f'{n} rows: pandas/Polars time ratios {ratios}; max parity error {max_error}', flush=True)
    Path(str(output_prefix)+'.json').write_text(json.dumps(report,indent=2)+'\n')
    lines = ['# Measured pandas versus Polars benchmark', '',
             'Model path: '+model_kind+'. Training input: '+report['method']['training_input']+'.', '',
             'Runtime: '+('warm in each worker' if warm_runtime else 'cold in each fresh worker')+'. Python '+platform.python_version()+'. Ratios above 1 favor Polars; below 1 favor pandas.', '',
             '| Rows | pandas E2E s (IQR) | Polars E2E s (IQR) | E2E ratio | Adapter ratio | pandas/Polars peak RSS MiB | Max parity error |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for item in report['datasets']:
        a,b = (item['summary'][k]['end_to_end'] for k in ('pandas','polars'))
        def timing(x):
            t=x['seconds']; return f"{t['median']:.4f} ({t['q25']:.4f}-{t['q75']:.4f})"
        ratios=item['pandas_time_over_polars_time']
        lines.append(f"| {item['rows']} | {timing(a)} | {timing(b)} | {ratios['end_to_end']:.3f} | {ratios['dataframe']:.3f} | {a['peak_rss_bytes']['median']/2**20:.1f}/{b['peak_rss_bytes']['median']/2**20:.1f} | {item['max_absolute_parity_error']:.3g} |")
    lines += ['', 'Native frame construction and model conversion/inference are included. Training and process startup are excluded.',
              'One fitted estimator, identical seeds, data, weights and settings per size; alternating backend order; one excluded warm-up per backend.',
              'RSS is sampled every 2 ms in isolated processes and is a lower bound on true peak. No free-threading speed claim is made.',
              'JSON includes every trial, inference time, adapter time, absolute and incremental RSS, dispersion, and package versions.']
    Path(str(output_prefix)+'.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', nargs='+', type=int, default=[1000,10000,100000])
    parser.add_argument('--features', type=int, default=10)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--warm-runtime', action='store_true')
    parser.add_argument('--output-prefix', default='dalex-narwhals-benchmark')
    parser.add_argument('--model', choices=['sklearn-numpy', 'sklearn-native', 'lightgbm-native'], default='sklearn-numpy')
    parser.add_argument('--sample-limit', type=int, default=10_000)
    parser.add_argument('--source-revision', default='unspecified')
    args = parser.parse_args()
    if args.sample_limit < 1:
        parser.error('--sample-limit must be positive')
    run(args.sizes,args.features,args.repetitions,args.rounds,args.warm_runtime,args.output_prefix,args.model,args.sample_limit,args.source_revision)
