# Backend integration testing

The backend migration includes analytical unit tests, real-model integration
tests, the existing DALEX suite, upstream numerical comparisons and isolated
benchmarks. These checks answer different questions; backend parity alone would
not catch a common error shared by both implementations.

## Run the complete suite

Install the package and CI requirements from the repository root:

```sh
python -m pip install -e ./python/dalex -r python/dalex/ci/requirements.txt
python python/dalex/ci/verify_runtime.py
python -m pytest -q -ra --timeout=120 --timeout-method=thread python/dalex/test
```

Use `MPLBACKEND=Agg` for headless execution. CI fixes model/BLAS/OpenMP threads
to one. Backend libraries and the real estimators are required dependencies of
the test environment; their integration cases do not silently skip when absent.
PPS runs separately with its upstream-compatible pandas<2 dependencies.

## Workflow coverage

| Test module | Integration boundary |
|---|---|
| `test_native_profile_workflows.py` | Fit sklearn/LightGBM regression and binary pipelines once on Polars; compare pandas/Polars/Arrow importance, contributions, CP-to-aggregate, diagnostics, performance, Aspect/triplots and fairness; exercise exports, plots, pickle and legacy mutations |
| `test_polars_output_pipelines.py` | Model-owned imputation/encoding and Polars-output preprocessing with sklearn/LightGBM prediction |
| `test_native_arena_workflows.py` | Model registration, all plot families, background threads/resources, incremental Shapley reductions, serialization and writable public resource dictionaries |
| `test_arena_missing_json.py` | HTTP custom observations and strict JSON with missing/nonfinite display values |
| `test_native_surrogate_workflows.py` | Real models through LIME and sklearn tree/linear surrogates, prediction adapters, exports, pickle and matplotlib plots |
| `test_native_label_workflows.py` | Custom labels across nine explanation methods, lazy native storage, original pandas index/column layouts and pickle |
| `test_native_estimators.py`, `test_native_tree_wrappers.py` | Polars-trained real estimators and external SHAP tree boundaries |
| `test_native_categorical_estimators.py`, `test_missing_native_estimators.py` | Explicit model-owned categorical/missing preprocessing and unsupported-input policies |
| `test_nullable_shap_boundaries.py`, `test_external_shap_seed.py` | Nullable numeric matrix adapters and sampled Kernel SHAP random-state behavior |
| `test_seeded_concurrency.py` | Serial/concurrent seeded explanation parity and caller RNG isolation |
| `test_native_datasets.py` | Every bundled CSV loader, parser parity, native backends and explicit row identifiers |
| `test_native_dendrogram_plotting.py` | Native base dendrogram fitting/plotting, tied/constant/missing data, edited metadata, real sklearn/LightGBM pipelines and pickle |
| `test_legacy_estimator_bridges.py` | Older LightGBM with an explicit model-owned matrix bridge across 13 explanation families, native exports, plots, edits and pickle |
| `test_pandas_feature_label_compatibility.py` | Working integer/MultiIndex/partial/missing pandas label contracts, exact label values and original extension dtypes |
| `test_float32_loss_compatibility.py` | Float32 and mixed-loss dtype/promotion, missing draws, exact pandas means, bounded native rounding and Python group labels |

`native_computation_only` guards reject pandas DataFrame/Series construction and
Narwhals-to-pandas conversion inside the guarded native computation. Native plot
preparation is also guarded; public legacy views are exercised separately. Thus
the tests check backend preservation and the compatible publication boundary.

Independent expected-value tests cover typed UInt64 grids above 2**63, exact
nanosecond values, finite Decimal interventions, native null versus NaN, missing/unused groups, mutation and
metadata. `test_native_profile_grid_scaling.py` also guards against exporting
every reference cell to Python merely to construct an automatic numeric grid.
Aspect/dendrogram tests reject browser-renderer calls and preserve the original
hover vertices and endpoints, including constant panels.
`ci/compare_legacy.py` compares numerical outputs with an unmodified
upstream checkout, using a documented NumPy spelling shim for the old reference.
`ci/check_legacy_pickles.py` loads real upstream explanation pickles in the new
package, checks their pandas tables, restores the documented Explainer dump
protocol and reuses old CP/Aspect objects. A required Python 3.14 CI job checks
both scripts against a pinned upstream revision.

## Runtime matrix and external libraries

`Python-check.yaml` runs the full suite on Python 3.10–3.14 on Linux/macOS,
Windows 3.14, pandas 3, NumPy 1.23.5/pandas 1.5.3, and combined Polars 1/Narwhals
2/Plotly 6 minimums. A Windows 3.10/LightGBM 4.6 bridge job covers older models.
The required
3.14t job checks the actual disabled GIL state after each dependency import.
An installed-wheel Python 3.14 job runs 12 core explanation/export/plot/pickle
workflows plus the base dendrogram with pandas 3 and without Polars, Arrow,
sklearn, SHAP, LIME or statsmodels.
Diagnostics explicitly disable optional smoothing; widget/estimator extras remain
separate capabilities. Wheel metadata verifies the documented dependency bounds.
Separate required CPU jobs exercise PyTorch and TensorFlow Deep/Gradient SHAP;
those tests live under `integration_tests/`, outside the ordinary test directory.

Forced-disabled-GIL passing tests do not certify arbitrary third-party compiled
extension concurrency. The isolated automatic-GIL probe records dependencies
that re-enable it without the override. Tested neural models are single-output
dense regressors; arbitrary architectures, GPUs, multiclass models, other
dataframe engines and lazy/distributed execution are not certified.

Benchmarks run separately from tests, using identical saved models, values,
seeds and explanation settings. Source/model fingerprints, every trial,
numerical errors, elapsed time and RSS belong in the report. A speed ratio is
not a substitute for integration correctness, and regressions remain visible.

## Native plotting follow-up

`test_native_plot_preparation.py` exercises native plot preparation for importance,
diagnostics, ECDF/ROC/lift, Break Down/interactions and DALEX Shapley, together with
real Tree/Kernel SHAP metadata fitting, exports, plotting inputs and serialization.
It guards legacy table access and conversions, compares all figure fields across
pandas/Polars/Arrow at 1e-12, and checks edited views, mixed feature metadata, large
integers, positional sampling and unequal/small lift datasets. KernelExplainer's
upstream local closure is not claimed to be pickleable.

`ci/compare_plots.py` independently compares 178 figures with pinned, unmodified
upstream: 168 complete payloads and six triplots/four base dendrograms after removing only redundant
unlabelled collinear hover samples. That density exception comes from removing
browser layout resolution; geometry, text and layout still match. This runs in
the required upstream-compatibility job and complements 29 numerical outputs,
including nonlinear hierarchical Aspect cases.

## Aggregated-profile plotting follow-up

`test_native_aggregate_plotting.py` adds 99 cases for partial, conditional and
accumulated line/bar figures, multiple models, categorical profiles, grouping,
mutable pandas views, exact large integer axes, missing/nullable numeric groups, nanosecond/timezone captions and native
fit-to-plot construction guards. The pandas-prepared raw CP overlay is exercised
for compatibility and is now covered by the CP native construction guarantee.
`ci/compare_plots.py` now also compares 30 profile figures with unchanged upstream
(83 comparisons before the CP follow-up; 132 after that checkpoint).

## CP plotting and publication dtypes

`test_native_cp_plotting.py` adds 91 cases covering complete native/published-view
figure equality, edited result/observation indexes, multiple feature schemas,
exact UInt64 marker membership, nanosecond/timezone/duration/date captions,
nullable and Arrow dtype publication, and strict native fit-to-plot guards.
Aggregated raw-profile overlays are included. The independent upstream harness
adds 49 CP figures, including numeric/categorical axes, multiple models and
observations, observation markers, colors and nullable floating captions.

## Fairness plots and legacy adapters

`test_native_fairness_plotting.py` adds 119 cases covering all classification and
regression fairness plot types, backend/published-view figure parity, native
construction guards, mutable score views, metric alignment, missing values,
cutoff setting snapshots, and historical private pandas adapter indexes/types.
Sixteen real fitted sklearn/LightGBM pipeline cases preserve original Polars or
Arrow model batches through fairness plotting. The upstream harness adds 20
complete fairness figures (152 overall) and fixes its child hash seed so legacy
radar trace ordering is comparable between processes.

## Aspect plots and consolidation of private adapters

`test_native_aspect_plotting.py` covers native/published-view figures, edited
tables, precise heterogeneous hover values, native widget callbacks, rounding
callback contracts and fit-time observation snapshots. `test_native_aspect_integration.py`
adds 48 real sklearn/LightGBM regression/classification pipeline cases from
native Polars/Arrow through fitting, exports, figures and pickle round trips.

`test_legacy_native_adapters.py` includes untouched-upstream golden regressions
for pandas index-sensitive callbacks and nonlinear hierarchical Shapley. The
latter exposed a serial RNG reset error concealed by linear models. It also
checks the optional PPS boundary independently of PPS installation: external
calls receive pandas, while score matrices return to the original backend.

`ci/compare_legacy_adapters.py` compares 265 outputs from an unmodified upstream
package in a separate interpreter. These cover private breakdown, Shapley, CP,
aggregate, categorical statistics and Aspect/triplot helpers, including indexes,
dtypes, nested metadata, serial/process paths and dtype-sensitive tie order.
The reference receives a documented NumPy spelling shim and a pickle-safe
residual callback. Matching global seed initialization permits comparison with
old helpers that relied on constructor-managed global NumPy state. The shared
kernels retain local explicit streams rather than resetting caller RNG state.

A 129-output label subset is repeated on pandas 3.1. It covers missing and
MultiIndex feature labels, partial selectors and the original runtime's
best-effort ordering. Float32 custom losses retain exact pandas reduction values
and dtypes. Native Float32 accumulators have a separate precision bound
(`rtol=8*2**-23`, `atol=2**-23`); 54 instrumented cases per full-suite job record
the observed errors. These are not represented as float64 or bitwise parity.

`test_duplicate_pandas_features.py` adds 79 cases for the duplicate-label APIs
that work upstream: importance, performance, diagnostics, fairness and LIME.
It covers grouped positional shuffles, extension dtypes, missing/reserved and
MultiIndex labels, output mutation/independence, old table state, custom labels,
process workers, external classes and typed matrix reconstruction/subsets.
The reference harness independently compares those supported contracts;
it does not turn inherited contribution/profile failures into a support claim.
`ci/check_legacy_pickles.py` loads 16 actual unchanged-upstream objects, including
a duplicate-column diagnostic table. Legacy pandas/native exports retain that
table; explicit Polars export rejects its ambiguous physical names.

The real optional PPS check covers pandas/Polars/Arrow, verifies native matrix
storage and compares numeric values in its pandas 1.5.3-compatible environment.
