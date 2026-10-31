"""Native fairness decisions with an explicit legacy console-format boundary."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from test_native_fairness_plotting import objects
from test_native_plot_preparation import no_legacy_materialization
from test_native_profile_workflows import native_computation_only


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
@pytest.mark.parametrize('regression', [False, True])
def test_native_fairness_thresholds_do_not_publish_pandas(monkeypatch, capsys, backend, regression):
    from dalex.fairness._group_fairness import utils
    value = objects(backend, regression)[0]
    seen = []

    def display(obj, frame, subgroups, metrics):
        seen.append((len(frame), list(metrics)))
        return SimpleNamespace(to_string=lambda: '<console display boundary>')

    monkeypatch.setattr(utils, '_fairness_check_display', display)
    with native_computation_only(monkeypatch, backend), no_legacy_materialization(monkeypatch):
        value.fairness_check(verbose=False)
    assert seen and '<console display boundary>' in capsys.readouterr().out
    assert '_compatibility_result' not in value.__dict__


@pytest.mark.parametrize('backend', ['polars', 'pyarrow'])
@pytest.mark.parametrize('regression', [False, True])
@pytest.mark.parametrize('edit', ['none', 'nan', 'nullable', 'infinite', 'duplicate'])
def test_fairness_console_matches_pandas_and_honors_edits(capsys, backend, regression, edit):
    pair = [objects(name, regression)[0] for name in ['pandas', backend]]
    texts = []
    for value in pair:
        if edit != 'none':
            metric = 'independence' if regression else 'TPR'
            frame = value.result
            frame.index.name = 'edited-group'
            if edit == 'nan':
                frame.loc['b', metric] = np.nan
            elif edit == 'nullable':
                frame[metric] = frame[metric].astype('Float64')
                frame.loc['b', metric] = pd.NA
            elif edit == 'infinite':
                frame.loc['b', metric] = np.inf
            else:
                value.result = pd.concat([frame.iloc[::-1], frame.loc[['b']]])
        value.fairness_check(epsilon=.85, verbose=True)
        texts.append(capsys.readouterr().out)
    assert texts[0] == texts[1]
    if edit == 'none':
        assert '_compatibility_result' not in pair[1].__dict__


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
def test_fairness_check_with_no_unprivileged_rows(capsys, backend):
    value = objects(backend)[0]
    value.protected = np.repeat(value.privileged, len(value.protected))
    value.fairness_check(verbose=False)
    assert 'No bias was detected!' in capsys.readouterr().out


@pytest.mark.parametrize('missing', ['row', 'metric'])
def test_fairness_check_preserves_legacy_missing_label_errors(missing):
    value = objects('polars')[0]
    value.result = value.result.drop(index='b') if missing == 'row' else value.result.drop(columns='TPR')
    with pytest.raises(KeyError):
        value.fairness_check(verbose=False)
