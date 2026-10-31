"""Exercise an installed wheel with core dependencies and no native extras."""
import argparse
import importlib.util
import importlib.metadata as metadata
import json
import hashlib
from pathlib import Path
import pickle
import sys


class Model:
    def predict(self, frame):
        return frame['x'].to_numpy() * 2 - frame['z'].to_numpy() / 2


def residual(model, frame, target):
    return target - model.predict(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assert-minimal', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    optional = {name: importlib.util.find_spec(name) is not None for name in ['polars', 'pyarrow', 'sklearn', 'shap', 'lime', 'statsmodels']}
    if args.assert_minimal:
        assert not any(optional.values()), optional
    import numpy as np
    import pandas as pd
    import dalex as dx
    source = Path(dx.__file__).resolve()
    assert 'site-packages' in source.parts, f'Expected installed wheel, imported {source}'
    from packaging.requirements import Requirement
    requirements = [Requirement(value) for value in metadata.requires('dalex')]
    bounds = {requirement.name: str(requirement.specifier) for requirement in requirements if requirement.marker is None}
    assert bounds['pandas'] == '>=1.5.3' and bounds['narwhals'] == '>=2.0', bounds
    assert 'polars' not in bounds and 'pyarrow' not in bounds
    frame = pd.DataFrame({'x': [1., 2., 3., 4., 5., 6.], 'z': [3., 1., 4., 2., 6., 5.]},
                           index=[10, 30, 20, 60, 50, 40])
    model = Model()
    target = model.predict(frame) + np.array([.1, -.1, .2, -.2, .1, -.1])
    exp = dx.Explainer(model, frame, target, residual_function=residual,
                       model_type='regression', verbose=False)
    np.testing.assert_allclose(exp.predict(frame), model.predict(frame))
    values = {
        'importance': exp.model_parts(N=None, B=2, random_state=42),
        'breakdown': exp.predict_parts(frame.iloc[[1]], type='break_down', N=None, random_state=42),
        'interactions': exp.predict_parts(frame.iloc[[1]], type='break_down_interactions', N=None, random_state=42),
        'shap': exp.predict_parts(frame.iloc[[1]], type='shap', N=None, B=2, random_state=42),
        'cp': exp.predict_profile(frame.iloc[[1, 3]], variables=['x'], grid_points=3, verbose=False),
        'performance': exp.model_performance(),
        'diagnostics': exp.model_diagnostics(),
    }
    for kind in ['partial', 'conditional', 'accumulated']:
        values[kind] = exp.model_profile(type=kind, variables=['x'], N=None,
                                         grid_points=3, random_state=42, verbose=False)
    aspect = dx.Aspect(exp)
    assert aspect.plot_dendrogram(show=False) is not None
    assert '_compatibility__dendrogram_aspects_ordered' not in aspect.__dict__
    assert pickle.loads(pickle.dumps(aspect)).plot_dendrogram(show=False) is not None
    values['model_aspect'] = aspect.model_parts(N=None, B=2, random_state=42)
    values['model_triplot'] = aspect.model_triplot(N=None, B=2, random_state=42)
    for name, value in values.items():
        assert isinstance(value.result, pd.DataFrame), name
        pd.testing.assert_frame_equal(value.result_as('pandas'), value.result)
        restored = pickle.loads(pickle.dumps(value))
        pd.testing.assert_frame_equal(restored.result, value.result)
        options = {'widget': False} if name == 'model_triplot' else {}
        if name == 'diagnostics':
            options['smooth'] = False  # Smoothing is an explicit statsmodels extra.
        assert restored.plot(show=False, **options) is not None, name
    # Public writable legacy results remain authoritative in this minimal install.
    values['importance'].result.loc[0, 'dropout_loss'] += .125
    pd.testing.assert_frame_equal(values['importance'].result_as('pandas'), values['importance'].result)
    files = sorted(source.parent.rglob('*.py'))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(source.parent).as_posix().encode() + b'\0')
        digest.update(path.read_bytes().replace(b'\r\n', b'\n') + b'\0')
    report = {'passed': True, 'python': sys.version, 'dalex_source': str(source),
              'current_source': {'python_modules': len(files), 'python_source_sha256': digest.hexdigest(),
                  'normalization': 'sorted relative paths and source bytes, CRLF normalized to LF'},
              'core_dependency_bounds': bounds,
              'base_dendrogram': True,
              'families': list(values), 'optional_modules_present': optional,
              'versions': {name: metadata.version(name) for name in ['dalex', 'pandas', 'numpy', 'narwhals', 'plotly', 'scipy']}}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
