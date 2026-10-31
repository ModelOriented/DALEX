"""Load real upstream pandas pickles in the migrated package, without editing it."""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile


def capture(root, output):
    sys.path.insert(0, str(Path(root).resolve()))
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression, LogisticRegression
    import dalex as dx
    assert Path(dx.__file__).resolve().is_relative_to(Path(root).resolve()), dx.__file__
    if not hasattr(np, 'in1d'):
        np.in1d = np.isin  # Execute the untouched upstream code on modern NumPy.
    rng = np.random.default_rng(123)
    frame = pd.DataFrame(rng.normal(size=(40, 3)), columns=['a', 'b', 'c'])
    y = frame.to_numpy() @ np.array([2., -1., .5])
    model = LinearRegression().fit(frame, y)
    exp = dx.Explainer(model, frame, y, label='linear', verbose=False)
    obs = frame.iloc[[2]]
    cp = exp.predict_profile(obs, variables=['a'], grid_points=5, verbose=False)
    aspect = dx.Aspect(exp)
    objects = {'explainer': exp, 'aspect': aspect, 'cp': cp,
        'importance': exp.model_parts(N=20, B=2, random_state=42),
        'breakdown': exp.predict_parts(obs, type='break_down'),
        'shap': exp.predict_parts(obs, N=20, B=2, random_state=42),
        'partial': dx.model_explanations.AggregatedProfiles(type='partial', variables=['a']),
        'performance': exp.model_performance(), 'diagnostics': exp.model_diagnostics(),
        'aspect_predict': aspect.predict_parts(obs, N=20, B=2, random_state=42),
        'aspect_model': aspect.model_parts(N=20, B=2, random_state=42),
        'predict_triplot': aspect.predict_triplot(obs, N=20, B=2, random_state=42),
        'model_triplot': aspect.model_triplot(N=20, B=2, random_state=42),
        'fairness_regression': exp.model_fairness(np.tile(['a','b'],20), 'a')}
    objects['partial'].fit(cp, verbose=False)
    binary = (y > np.median(y)).astype(int)
    classifier = LogisticRegression().fit(frame, binary)
    classification = dx.Explainer(classifier, frame, binary, label='binary', verbose=False)
    objects['fairness_classification'] = classification.model_fairness(np.tile(['a','b'],20), 'a')
    # This real old state cannot be wrapped as a Narwhals frame: its feature
    # labels are duplicated. Only the explanation table is pickled, so the
    # local predictor closure is not part of this object's serialized state.
    duplicate = frame.copy()
    duplicate.columns = ['a', 'a', 'c']
    duplicate_exp = dx.Explainer(model, duplicate, y, model_type='regression',
        predict_function=lambda fitted, batch: fitted.predict(np.asarray(batch)),
        label='duplicate', verbose=False)
    objects['diagnostics_duplicate'] = duplicate_exp.model_diagnostics()
    # Use DALEX's documented dump protocol to drop its default local residual
    # callback. A raw pickle of that callback already fails in upstream.
    explainer_bytes = exp.dumps()
    objects['explainer'] = pickle.loads(explainer_bytes)
    aspect.explainer = objects['explainer']
    payload = {'objects': objects,
        'tables': {name: obj.result.copy() for name,obj in objects.items() if name not in ('explainer','aspect')},
        'depend_matrix': aspect.depend_matrix.copy(), 'groups': aspect.get_aspects(),
        'explainer_bytes': explainer_bytes}
    Path(output).write_bytes(pickle.dumps(payload, protocol=5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-root', required=True)
    parser.add_argument('--output', default='legacy-pickle-compatibility.json')
    parser.add_argument('--capture', action='store_true')
    args = parser.parse_args()
    if args.capture:
        capture(args.reference_root, args.output)
        return
    import numpy as np
    import pandas as pd
    import dalex as dx
    with tempfile.TemporaryDirectory(prefix='dalex-legacy-pickle-') as folder:
        path = Path(folder)/'upstream.pkl'
        subprocess.run([sys.executable, __file__, '--capture', '--reference-root', args.reference_root,
                         '--output', str(path)], env=os.environ.copy(), check=True)
        payload = pickle.loads(path.read_bytes())
    objects = payload['objects']
    objects['explainer'] = dx.Explainer.loads(payload['explainer_bytes'])
    for name, expected in payload['tables'].items():
        pd.testing.assert_frame_equal(objects[name].result, expected)
        pd.testing.assert_frame_equal(objects[name].result_as('pandas'), expected)
        if not expected.columns.is_unique:
            pd.testing.assert_frame_equal(objects[name].result_as(), expected)
            try:
                objects[name].result_as('polars')
            except ValueError as error:
                assert 'Duplicate pandas column labels' in str(error)
            else:
                raise AssertionError('Native exports must reject ambiguous duplicate columns')
    pd.testing.assert_frame_equal(objects['aspect'].depend_matrix, payload['depend_matrix'])
    assert objects['aspect'].get_aspects() == payload['groups']
    assert objects['aspect'].plot_dendrogram(show=False) is not None
    aggregate = dx.model_explanations.AggregatedProfiles(type='partial', variables=['a'])
    aggregate.fit(objects['cp'], verbose=False)
    pd.testing.assert_frame_equal(aggregate.result, payload['tables']['partial'], check_exact=False,
                                  rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(objects['explainer'].predict(objects['explainer'].data), objects['explainer'].y_hat)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
    from _benchmark_utils import source_identity
    result = {'passed':True, 'reference_root':str(Path(args.reference_root).resolve()),
        'objects_loaded':len(objects),'legacy_tables_compared':len(payload['tables']),
        'current_source':source_identity(dx),'legacy_duplicate_pandas_exports':True,
        'legacy_cp_aggregation':True,'legacy_aspect_groups_and_plot':True,
        'python':sys.version,'pandas':pd.__version__}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
