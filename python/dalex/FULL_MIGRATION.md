# Full Narwhals migration: implementation contract and progress

This defines the expanded migration following the native-feature contribution.
The [latest source-stamped review](benchmarks/results/NARWHALS_FINAL_REVIEW.md) records
the validated implementation, benchmarks and remaining boundaries. Earlier
reports do not certify later commits automatically. PR #587 remains draft for
manual review.

The latest verified stable pandas release is **3.1.0 (October 22, 2026)**,
checked against the [official release notes](https://pandas.pydata.org/docs/whatsnew/index.html)
and [PyPI release metadata](https://pypi.org/pypi/pandas/json) on October 31, 2026.
There is no published pandas 4 release. Speculative future-major pandas and
Narwhals package caps have been removed; code selects the maintained Narwhals
stable.v2 API. Existing development-build tests use pandas 3.2, not pandas 4.
Only actual tested versions are evidence of compatibility. CI major-version
selectors are reproducible test choices rather than runtime package limits.

## Compatibility contract

- Existing pandas inputs, default return types, column labels, indexes, plotting
  methods and writable `.result` tables continue to work. Native feature batches
  must not become pandas on their way to a predictor.
- Dataframe computation uses `narwhals.stable.v2`; numerical vector algorithms
  may use NumPy/SciPy. Explicit estimator/tensor and serialization boundaries
  remain separate from dataframe manipulation.
- Public pandas result tables can be materialized lazily. Once read, their
  mutations are authoritative for subsequent exports and serialization.
- `explanation.result_as()` exports an independent table in its computation
  backend. `result_as('pandas'|'polars'|'pyarrow')` requests an explicit backend.
  `table='permutation'` exports raw importance draws. SHAP's non-tabular arrays
  retain their existing representation and support only the native export.
- Fairness auxiliary score exports retain explicit `subgroup` or `metric`
  labels in native tables. The legacy `parity_loss` attribute and its explicit
  pandas export remain a Series, with caller edits respected on later exports.
- Arena registration, attributes, EDA, profiles, metrics, fairness and background
  Shapley resources use native tables. Resource dictionaries still publish pandas
  values lazily; `resource.result_as(partition=feature)` exports typed draws.
- Dataset loaders accept `backend='polars'|'pyarrow'`, defaulting to pandas.
  Native defaults contain the historical feature columns; `include_index=True`
  retains CSV row identifiers as an explicit column instead of a hidden index.
- CP's native `result_as(partition=variable)` retains each original feature's
  scalar type. Whole-table native exports reject partitions with incompatible
  scalar types or unsafe integer promotion instead of silently stringifying or
  rounding. The combined, writable legacy pandas table remains available.
- Native grouped-profile tables retain typed grouping columns; the legacy
  pandas projection encodes those memberships into its existing string labels.
  Once a pandas projection is materialized it is authoritative, including its
  column layout, for subsequent exports.
- Arbitrary Python/non-string labels that cannot be physical native column names
  remain at an explicit pandas compatibility boundary, rather than being changed
  to strings silently. Default installation behavior remains compatible; moving
  pandas out of required dependencies is a separate packaging decision until
  no-pandas runtime coverage and default-output behavior have been demonstrated.

## Evidence and completion gates

The review report records the evidence for these criteria and its exact scope:

1. Native storage and reductions for importance, contributions, profiles,
   performance/diagnostics, Aspect, fairness and Arena.
2. Plotting/data preparation and dataset loaders use explicit backend boundaries.
3. Compatible pandas results and native exports are tested for missing values,
   integer precision, mixed/temporal data, labels, indexes and mutation.
4. Real fitted sklearn/LightGBM preprocessing pipelines run end-to-end across
   pandas, Polars and Arrow, with conversion guards on native computation.
5. CP-to-aggregate, explanation-to-plot, serialization and Arena/fairness/Aspect
   integration tests complement analytical unit tests and upstream comparisons.
6. The complete suite, minimum dependencies, current/nightly pandas, installed
   wheel, neural jobs and actual-disabled-GIL 3.14t CI are revalidated.
7. Benchmarks identify exact source/model fingerprints and retain performance,
   memory and numerical regressions as well as improvements.
8. A final inventory classifies remaining pandas operations as public
   compatibility, external dependency boundaries or unfinished migration work.

## Current implementation

- Shared result-table storage supports lazy legacy pandas views, explicit native
  exports, in-place pandas edits, independent exports and old pickle state.
- Permutation repetitions and their means run through Narwhals, returning native
  result/raw tables until their legacy public attributes are requested.
- CP prediction/profile-grid construction and partial, conditional and
  accumulated-profile computation run through typed native partitions. Group
  display labels and legacy row indexes are formatted at the pandas boundary.
- Analytical kernel tests and real sklearn/LightGBM pipeline integrations cover
  native model-to-CP-to-aggregate flows, missing groups, native exports, legacy
  result mutation, pickle, plots and nanosecond categorical levels. Guards reject
  pandas DataFrame/Series construction during native computation.
- Performance metrics and residual diagnostics retain native result/residual
  tables; the pandas performance index remains compatible and native metrics
  include an explicit model label. ROC-AUC is a vector reduction with tie and
  large-integer-score tests, rather than a pandas grouping pipeline.
- Breakdown, interactions and DALEX Shapley use native feature edits and native
  contribution/distribution tables. Shapley original-value metadata stays typed;
  use a variable partition for native exports where scalar types cannot be
  combined safely. Legacy result tables and seeded path behavior are retained.
- Aggregated profiles retain exact large-integer grids and separate native
  null/NaN bins and memberships. Mixed unsafe scalar layouts use explicit typed
  partitions, with a compatible combined pandas publication.
- Decimal CP interventions retain Decimal values. Automatic numeric grids keep
  legacy floating interpolation; supply explicit integer splits for models that
  depend on exact large-integer counterfactuals.
- Arena integration tests cover real models through registration, background
  threads, incremental updates, resource/plot exports, exact timestamp metadata,
  legacy dictionary access and mutation, pickle and strict JSON. EDA handles
  all-missing samples and unused categories without altering model data.
- CSV parser parity tests cover every bundled dataset, optional row identifiers
  and native loading with pandas construction disabled. The minimum dependency
  combination and pandas nightly are tested separately from the primary suite.
- Classification/regression fairness use vector metrics and native result
  tables; subgroup indexes and the pandas parity-loss Series remain compatible.
  Real-model integrations include lazy predictions, native exports, mutation,
  pickle, fairness checks and plots. Numerical comparisons also exercise an
  unmodified upstream implementation of both fairness families.
- Aspect association statistics, interventions, importance reductions and
  hierarchical triplot construction compute without pandas on native inputs.
  Dependence matrices and auxiliary tables retain lazy legacy views. Native
  Aspect `variable_values` use typed structs keyed by feature name, preserving
  mixed scalar values without converting numbers or timestamps to text.
- Real sklearn/LightGBM integrations cover Aspect/triplots, exports, writable
  pandas results, caches, pickle and plots. Tied heights now retain every merge;
  constant/all-missing associations normalize consistently. Cache reuse checks
  the random seed and process count as well as explanation settings.
- Native LIME and sklearn surrogate workflows retain lazy result/performance
  views, independent exports, plots and pickle. Pandas inputs retain the exact
  original external classes. External fitting/explanation algorithms use their
  explicit numeric matrix boundary.
- Custom explanation labels update native storage, including the historical
  performance index/added-column layout. Numeric profile grids avoid copying
  clean reference columns, and CP normalizes split values once per grid.
- External SHAP observation metadata stays native until its legacy attribute is
  read. `result_as(table='new_observation')` exports it independently. SHAP plots
  use numerical arrays without publishing that pandas view, and explicit edits
  to an already-read view remain authoritative.
- Importance, residual-diagnostic, performance (ECDF/ROC/lift), Break Down and
  DALEX Shapley plot preparation now uses native tables and numerical vectors.
  Plotting does not implicitly publish their legacy result tables. Mixed Shapley
  feature metadata need not be flattened to draw contribution bars. The upstream
  compatibility CI job compares complete plot payloads, including tie order,
  sampling, hover text and layout. Lift plots also handle fewer than 101 rows and
  models with different row counts.
- CP, aggregate overlays, all fairness figures and Aspect/triplot plot preparation
  retain native tables. Typed metadata and caller edits remain compatible. Native
  widget callbacks, real fitted pipelines and observation snapshots are tested.
- Older breakdown/Shapley, CP/aggregate and Aspect/triplot helpers now adapt
  shared native kernels, retaining their historical pandas return contracts.
  Categorical statistics share vector kernels; PPS score reshaping returns to
  the original backend after its unavoidable external pandas call. Independent
  upstream checks cover 265 helper/public/text outputs, including indexes and
  dtypes. A 129-output label subset is repeated independently on pandas 3.
- Nonlinear hierarchical Shapley advances one serial RNG stream across cuts
  and final singles, preserving upstream estimates. A six-feature regression
  case exposed a reset error missed by the earlier linear examples. Pandas
  callbacks also retain historical Aspect intervention-index handling.

- Fairness text checks now select/order native score tables and calculate
  thresholds on detached numeric vectors. Their historical pandas console
  renderer is an explicit output boundary; caller-edited nullable dtypes retain
  their existing decisions and formatting. Independent upstream text comparisons
  complement native construction guards.
- Base Aspect dendrogram tooltips also read native metadata directly. Their
  plotting path respects explicit legacy metadata edits without publishing an
  otherwise unread pandas view. Native construction guards, real fitted
  pipelines and pickle round trips cover this route.
- Pandas MultiIndex feature labels, partial level selectors and missing labels
  retain their working legacy importance/profile contracts. Original label
  values and runtime-specific best-effort ordering remain at the pandas
  metadata boundary. Grouped extension columns retain their input dtypes.
- Float32 custom losses retain their public dtype and pandas reduction rounding.
  Mixed Float32/Float64 loss lists are promoted explicitly before backend
  construction; nullable draws retain native nulls. Native Float32 accumulator
  differences are tested with a precision-appropriate bound and reported
  separately from float64 parity.
- Working duplicate-label pandas importance, performance, diagnostics, fairness
  and LIME routes remain supported. Permutations select physical columns jointly;
  diagnostic vectors share the native kernel while original labels are published
  separately. External matrix reconstruction retains physical positions, column
  Index/MultiIndex metadata, categorical dtypes and the legacy RangeIndex. These
  inputs already fail in upstream contributions/profiles, so those methods still
  require unique feature labels. Duplicate output columns must be renamed before
  exporting to Polars/Arrow.
  Diagnostic plotting through a Narwhals frame likewise needs uniquely named
  selected variables or a renamed writable result projection.

## Remaining validation and precision boundaries

- Combining incompatible native partition scalar types remains a precision
  boundary. Do not silently round large integer axes to float. Existing guards
  and cross-backend tests certify covered workflows, not every possible callback,
  dtype or third-party dependency.
- Performance remains workload-dependent. The latest report separates input
  conversion, initialization and explanation time, default precalculation,
  thread settings, absolute/incremental RSS and regressions. Earlier plotting
  measurements do not supersede end-to-end explanation benchmarks.

Pandas-free installation, lazy execution and untested dataframe engines remain
separate scope. NumPy/SciPy/tensor boundaries are intentional numerical contracts;
SHAP's automatic GIL fallback and PPS dependency limits are external issues.

## External migration precedents

[Plotly 6](https://plotly.com/python/v6-migration/) adopted Narwhals for native
pandas/Polars/Arrow dataframe processing. Its change illustrates using a shared
internal dataframe layer while retaining existing pandas usage.

[Pandera's Narwhals backend](https://pandera.readthedocs.io/en/latest/narwhals_backend.html)
keeps its existing public imports and offers opt-in backend activation. This
supports separating internal migration from changes to a library's public API;
DALEX's specific API and numerical semantics still require its own tests.

[LightGBM 4.7](https://github.com/lightgbm-org/LightGBM/releases/tag/v4.7.0)
introduced Polars input through Narwhals and the Arrow PyCapsule interface.
DALEX preserves those native inputs at the model boundary. Older estimators can
instead use an explicit model-owned numerical bridge; that route has separate
integration coverage and does not imply support for every dependency combination.

[Narwhals' stable API policy](https://narwhals-dev.github.io/narwhals/backcompat/)
provides a versioned namespace for compatibility. The declared 2.0 minimum is
tested explicitly; newer convenience methods cannot be assumed available there.

## CP plotting follow-up

CP line/bar preparation and aggregated raw-profile overlays retain native tables.
Observation indexes align original predictions before multiple-model concatenation.
Typed axes and exact marker comparisons stay separate from scalar hover formatting.
Explicit pandas result and observation edits remain authoritative. A shared column
formatter preserves historical pandas publication dtypes without the deprecated
`convert_floating` flag; this is a display compatibility boundary. Native CP
fit-to-plot guards forbid pandas DataFrame/Series construction.

The focused suite covers 91 cases, and the required upstream comparison now
includes 132 full figures. These checks do not certify every possible mixed
scalar partition: incompatible exact large-integer/float axes still require
separate selections rather than silent coercion. The subsequent fairness and
Aspect plotting reports record their own source-stamped evidence; their counts
do not retroactively apply to this CP checkpoint.

## Fairness plotting follow-up

Classification fairness checks, metric scores, stacked/radar, performance/fairness,
heatmap, cutoff and density plots use native preparation. Regression fairness
checks and density plots are covered as well. Native score tables preserve
subgroup/metric order and explicit pandas edits. The legacy vertical/horizontal
confusion tables and private parity/ratio helpers are adapters over shared native
kernels, retaining their pandas return types and indexes.

Intentional corrections: cutoff plotting copies settings instead of mutating
callers; heatmaps align metrics by name and leave absent values missing; plots
use finite bounds when scores are missing; cutoff searches report an actionable
error when no requested-metric total is finite. Metric-score plots honor explicit
public table edits. Default finite figures are independently compared with
unchanged upstream. Arbitrary user-assigned score dtypes remain uncertified.
