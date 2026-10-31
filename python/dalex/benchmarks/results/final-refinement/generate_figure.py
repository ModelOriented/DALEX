"""Render the retained original-Polars benchmarks, with their actual IQRs.

Run after extracting/checking no additional files: measurements.zip and its
manifest are read directly. Requires matplotlib and numpy; performs no models.
"""
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'benchmark-manifest.json').read_text(encoding='utf-8'))
entries = {entry['filename']: entry for entry in manifest['reports']}
archive_bytes = (root / manifest['archive']['filename']).read_bytes()
assert hashlib.sha256(archive_bytes).hexdigest() == manifest['archive']['sha256']
reports = {}
with zipfile.ZipFile(root / manifest['archive']['filename']) as archive:
    for scenario in ['million', 'default']:
        for model in ['sklearn', 'lightgbm']:
            name = f'final-{scenario}-{model}.json'
            content = archive.read(name)
            assert hashlib.sha256(content).hexdigest() == entries[name]['sha256']
            report = json.loads(content)
            assert report['passed']
            reports[scenario, model] = report

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.spines.left': False, 'axes.axisbelow': True,
                     'axes.titlesize': 11, 'axes.titleweight': 'bold'})
fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4))
series = [('upstream', 'pandas', 'Upstream pandas', '#64748b'),
          ('candidate', 'pandas', 'New pandas', '#d97706'),
          ('candidate', 'polars', 'New Polars', '#0f766e')]
captions = {'importance': 'Importance', 'breakdown': 'Break Down', 'shap': 'DALEX Shapley',
            'partial': 'Partial', 'conditional': 'Conditional', 'accumulated': 'Accumulated'}
for row, model in enumerate(['sklearn', 'lightgbm']):
    for column, scenario in enumerate(['million', 'default']):
        ax = axes[row, column]
        report = reports[scenario, model]
        families = report['settings']['families']
        positions = np.arange(len(families))
        for i, (source, backend, caption, color) in enumerate(series):
            stats = [next(item['seconds'] for item in report['summaries']
                          if (item['family'], item['source'], item['backend']) ==
                          (family, source, backend)) for family in families]
            medians = np.array([item['median'] * 1000 for item in stats])
            errors = np.array([[item['median'] - item['q25'] for item in stats],
                               [item['q75'] - item['median'] for item in stats]]) * 1000
            ax.barh(positions + (i-1)*.24, medians, height=.22, color=color, label=caption,
                    xerr=errors, error_kw={'elinewidth': .8, 'capsize': 2, 'ecolor': '#334155'})
        ax.set_yticks(positions, [captions[family] for family in families])
        ax.invert_yaxis()
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color='#e2e8f0', linewidth=.7)
        ax.set_xlim(left=0)
        ax.set_xlabel('Whole workflow, milliseconds')
        name = 'sklearn HistGradientBoostingRegressor' if model == 'sklearn' else 'LightGBM LGBMRegressor'
        setting = 'precalculate=False' if scenario == 'million' else 'normal default: precalculate=True'
        ax.set_title(name + '\n' + setting, loc='left', pad=12)

fig.suptitle('Synthetic regression workflows from an existing Polars frame',
             x=.075, y=.975, ha='left', fontsize=15, fontweight='bold')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.068, .935),
           ncol=3, frameon=False, fontsize=10)
fig.subplots_adjust(left=.12, right=.98, bottom=.23, top=.84, hspace=.55, wspace=.42)
fig.text(.075, .035,
         'Reference: 1m rows × 10 Float64 features; explanation N=1,000, B=3 where applicable. Bars: median; whiskers: IQR.\n'
         'Conversion, new Explainer, explanation and pandas publication included; training/imports/input creation excluded.\n'
         'Ordinary Python 3.14.4; model/BLAS threads=1, Polars=8. Panels have different scales. RSS tradeoffs are in the report.',
         ha='left', va='bottom', fontsize=9, color='#475569', linespacing=1.55)
output = root / 'whole-workflow-times.png'
fig.savefig(output, dpi=160, facecolor='white')
plt.close(fig)
print(output)
