# Full Narwhals migration review — 30 October 2026

Historical checkpoint: the [October 31 final review](NARWHALS_FINAL_REVIEW.md)
records the subsequent implementation, compatibility fixes and fresh benchmarks.
The measurements below belong only to the source identified here.

The supported eager pandas/Polars/Arrow explanation families now compute on native
tables, with the existing writable pandas API retained at explicit compatibility
and presentation boundaries. The contribution is ready for the author's and
maintainers' review in **draft PR [#587](https://github.com/ModelOriented/DALEX/pull/587)**.
It remains draft and unmerged. This certifies the tested scope below; arbitrary
models, dtypes, engines and dependency combinations are not implied.

Validated library source: [080f946b](https://github.com/brunocarlin/DALEX/commit/080f946b4bb3f53134957875d8e7b3fbd854d71c).
Library Git tree: `512d02f10fe29440d8ecfc2aea8fece9ebff634b`. Documentation/report successors can
share this tree; historical reports are never reassigned to new implementation
commits. [Machine-readable validation](full-migration/validation.json) and
[source/model benchmark manifest](full-migration/benchmark-manifest.json) retain
the exact evidence.

## Implemented scope

- Native feature edits and result storage/reductions cover permutation importance,
  breakdown/interactions, DALEX Shapley, CP, partial/conditional/accumulated
  profiles, metrics/diagnostics, Aspect associations/importance/triplots,
  classification/regression fairness, Arena EDA/plot resources and all bundled
  dataset loaders. Numerical vector operations use NumPy/SciPy.
- Native `result_as()` exports coexist with lazy, writable legacy pandas tables.
  Exports support explicit backends, independent table edits, original indexes,
  old pandas object state and nested non-tabular SHAP array copies. Reading a
  public pandas view makes its edits and layout authoritative thereafter.
- Typed feature partitions retain nanosecond metadata, large UInt64 values,
  native null versus NaN and mixed observation values. Unsafe combined scalar
  promotions require a partition. Profiles distinguish native missing bins and
  groups; numeric accumulated effects require a finite, ordered grid.
  Finite Decimal counterfactuals retain exact Decimal values across backends.
- Native LIME and sklearn surrogate results retain their external APIs, exports,
  pickle and plots. Pandas inputs retain the exact original external return
  classes. External estimator/SHAP/LIME tensor or matrix algorithms have explicit
  numerical boundaries.
- Custom labels across nine explanation methods update native storage; the
  historical performance index is retained when an override adds a label column.
- Native Arena background resources retain typed snapshots, incremental Shapley
  statistics, lazy public dictionaries, mutation, pickle and strict JSON.
  Dataset loaders default to their original pandas behavior; native backends
  optionally retain CSV row identifiers as explicit columns.
- Base Aspect dendrograms and triplots construct hover coordinates without a
  browser renderer. Vertex/endpoint and constant-panel tests guard this path.

## Validation

[All **21 fork CI jobs pass**](https://github.com/brunocarlin/DALEX/actions/runs/41141134799) on the implementation commit. The downloaded
artifacts were checked rather than relying on the workflow badge: 18 JUnit
reports contain 15 complete-suite configurations, two neural jobs and the legacy
PPS suite. Six CI cold/warm benchmark reports have zero measured parity error.
The upstream workflow's separate approval gate is a maintainer-controlled state,
rather than evidence that upstream has approved or executed the fork checks.

| Local configuration | Passed | Skipped | Failures/errors |
|---|---:|---:|---:|
| Python 3.14.4; pandas 2.3.3; Polars 2.1; Narwhals 2.28 | 1215 | 18 | 0 |
| Polars 1.0 / Narwhals 2.0; same ordinary Python/pandas | 1215 | 18 | 0 |
| Available pandas 3.2 development wheel | 1215 | 18 | 0 |
| pandas 3.1 / Narwhals 2.0 selected integration suite | 469 | 0 | 0 |
| Built wheel, imported from its extracted package | 252 | 0 | 0 |

The 18 skips are explicitly gated legacy PPS cases. A required Python 3.10,
pandas 1.5, ppscore-compatible CI job runs all 18 separately with no skips.
Required CPU neural jobs pass eight PyTorch cases on Python 3.14 and eight
TensorFlow cases on Python 3.13. Full-suite CI covers ordinary Python 3.10–3.14
on Linux/macOS, Windows 3.14, pandas 3, minimum pandas, minimum Polars/Narwhals,
and Linux 3.14t. Warnings remain visible; notably older Narwhals emits pandas 3
deprecations in the selected compatibility run.

Real sklearn/LightGBM regression and binary pipelines are fitted once on Polars
and exercised through all three dataframe backends. Integrations cover
preprocessing, explanations, exports, plots, writable legacy views, pickle,
Arena/background threads, fairness and Aspect lifecycles. Native computation
guards reject pandas construction and Narwhals-to-pandas conversion. Analytical
expected values and upstream comparisons complement backend agreement.
See the [integration test map](../../test/INTEGRATION_TESTING.md).

All **27 upstream numerical outputs** pass at rtol=atol=1e-12. Maximum absolute
error is **4.44e-16 locally** and
**8.88e-16 in CI**.
The untouched reference uses a documented `np.in1d`/`np.isin` spelling shim for
modern NumPy. Fifteen actual upstream objects load in the migrated package;
13 legacy pandas tables match, the documented Explainer dump/load protocol
restores callbacks, and old CP/Aspect objects remain usable. These checks run
in the required pinned-upstream CI job as well as locally.

Intentional corrections to broken or undefined legacy edge cases are separate
from those finite reference outputs: observed missing groups are retained,
unused empty groups are omitted, categorical conditional labels match numeric
or boolean values rather than producing zero effects, and tied-height triplots
retain every merge. Cache keys include seeds/process counts. Independent
expected-value tests cover these changes; backwards compatibility does not mean
preserving those erroneous outputs. Hover point density can differ while the
original vertices/endpoints are preserved.

### Free-threaded runtime

The 3.14t complete suite passes **1215/18**, with `Py_GIL_DISABLED=True` and
actual GIL state **False after all 21 dependency imports**. The committed
conda-forge/pixi route supplies the Polars free-threaded runtime unavailable
through the required PyPI wheel route. Runtime logs and the isolated automatic
GIL probe are archived in `full-migration/`.

Without `PYTHON_GIL=0`, SHAP 0.54.1 enables the GIL in the isolated probe. That
upstream extension limitation is separate from DALEX assertion failures. Passing
forced-disabled tests does not establish arbitrary concurrent compiled-extension
safety. Extension safety flags are not patched, and no free-threading speedup is
claimed. Seeded concurrency guarantees cover the tested cooperating RNG/model
paths; arbitrary external RNG consumers and model options remain outside them.

## Benchmarks and regressions

[Full tables, every trial, quartiles and RSS](FULL_MIGRATION_BENCHMARKS.md) use
identical saved models/data/seeds/settings. Ordinary Windows Python 3.14; five
trials/backend; isolated alternating workers warmed internally; model/BLAS
threads one and Polars eight. Original Polars creation/training/startup are
excluded, conversion to pandas is included, and both routes publish pandas
explanation results. No other DALEX tests or benchmarks ran during the timings.
Desktop background activity and fixed source-version ordering limit inference.

For one million original Polars rows, ten features and 1,000 explained rows,
pandas/Polars total-time ratios span **1.595–2.716**.
For 128 rows/1,000 features, breakdown/Shapley ratios span
**1.527–3.785**. Above one favors Polars.
Maximum measured numerical differences are **8.88e-16**
and **0**, respectively (rtol=atol=1e-10).

Some explanation-only stages regress despite avoiding full-frame conversion.
The report also compares current pandas with unmodified upstream pandas using
the same saved estimators: candidate/upstream total-time ratios span
**0.644–1.245**;
ratios above one are regressions. Absolute and incremental RSS, including wide
Polars memory regressions, remain visible. These observations establish neither
universal speedups nor a causal improvement from small between-session changes.

The benchmark found a substantial automatic-grid regression: Python scalar
filtering scanned the full reference column before explaining a small sample.
Native filtering/range reductions replace it, with a regression guard against
reintroducing full scalar exports. The earlier source-650a11fe measurement is
retained separately rather than relabeled as final evidence.
Profiling then identified redundant clean-column copies and repeated expanded
split checks. Those were removed in source 080f946b, with fresh five-trial
measurements; source-783f0697 figures remain separately stamped. Remaining pandas
regressions are still reported rather than averaged away.

## Resolved failures during this refinement

- Minimum Narwhals/modern Polars Enum-category metadata used a removed backend
  method; explicit enum metadata and a categorical fallback restore the matrix.
- An Arena HTTP conversion guard patched a removed, unused alias. It now guards
  the Narwhals conversion itself rather than restoring dead production code.
- Missing-value profile tests caught pandas-origin NaN grids against native
  null data. A sole missing state matches compatible spellings; genuinely
  distinct native null/NaN bins remain separate.
- Native custom-label tests exposed eager pandas publication; label updates
  now retain native tables and the exact legacy index/column layout.
- A local minimum-dependency run hung in an inherited base-Aspect browser
  rendering call. Numeric hover geometry and renderer-forbidden tests remove
  that dependency. No skip or increased timeout hides the failure.
- Earlier local runs were affected by a missing headless matplotlib setting or
  a full C: volume preventing report writes. Those runs are not counted as
  certified passes. Final local suites, wheel checks and benchmarks use the
  D: validation checkout and temporary storage; all reported XML is complete.

## Remaining boundaries and maintainer decisions

1. Pandas remains required for default writable tables, plot preparation, index
   and extension-dtype compatibility, scalar display formatting and optional
   PPS. A pandas-free distribution or replacement public result API requires a
   separate compatibility decision. Private pandas reference/compatibility
   algorithms remain; the public fit paths use the native kernels.
2. External matrix/tensor methods can coalesce null/NaN or round large integers
   when their library requires floating data. Those models need an explicit
   training/prediction contract. Core native attribution/profile precision
   coverage does not certify every external algorithm or object cell.
3. Raw LightGBM dictionary/Enum and pandas ArrowDtype cases require the documented
   model-owned encoding/bridge. Older sklearn with modern Polars needs the
   tested preprocessing bridge. Observation frames follow the explainer's
   backend and ordered feature names. [Recipes](../../NATIVE_BACKEND_RECIPES.md).
4. Lazy/distributed/GPU frames, other eager engines, arbitrary Python object
   columns, arbitrary model concurrency, multiclass and complex multi-input
   neural architectures are not certified. Numeric ALE missing splits remain
   an explicit error because missing values have no numeric ordering. Automatic
   uniform/quantile grids retain legacy floating interpolation; use explicit
   integer splits when a model depends on exact large-integer interventions.
5. Python **>=3.10**, dropping 3.9, and Narwhals **>=2,<3** need maintainer review.
   Pandas remains **>=1.5.3,<4**. The <4 cap is precautionary; no pandas 4 package
   was available when checked on 2026-10-30. Available pandas 3.2 development
   coverage does not certify pandas 4. Official [releases](https://pypi.org/project/pandas/)
   and [release notes](https://pandas.pydata.org/docs/whatsnew/).
6. SHAP's automatic GIL fallback requires upstream extension work, rather than
   silently changing DALEX or declaring that extension safe. PPS retains its
   separate older-pandas dependency environment. Current required jobs pass;
   these dependency constraints are documented separately from regressions.
7. Performance tuning remains worthwhile for the pandas regressions and wide
   native memory cases shown in the tables. Maintainers should evaluate those
   tradeoffs against the supported native workflows, rather than infer that
   replacing pandas accelerates arbitrary estimator inference.

Upstream/master was verified at `77ea3ad9580a3b68aaa94d8dca75d11bd527848f` and
remains unchanged. The fork branch is pushed; PR #587 remains draft and unmerged
for manual review. No reviewer request, maintainer comment or merge is made.
AI assistance is disclosed without implying maintainer approval of AI-authored
contributions.

## Implementation commits

| Commit | Change |
|---|---|
| [389278f8](https://github.com/brunocarlin/DALEX/commit/389278f8edd2458e300cc10360aa9bd647a12919) | Native importance/result storage |
| [c562a191](https://github.com/brunocarlin/DALEX/commit/c562a191b7f33286121e4891aaad7b93d2666d5a) | Typed CP and native aggregated profiles |
| [9bc4c134](https://github.com/brunocarlin/DALEX/commit/9bc4c1347a8b2ffba66f2f98630b235515ef9730) | Native contributions, metrics and diagnostics |
| [fcea8e7b](https://github.com/brunocarlin/DALEX/commit/fcea8e7b290342371ba193fc2f5c8150803c02b5) | Native Aspect/triplots and fairness |
| [391bdc93](https://github.com/brunocarlin/DALEX/commit/391bdc936d1c32ae15d62aaf115611dfce341201) | Older Narwhals/modern Polars Enum compatibility |
| [64ea4b14](https://github.com/brunocarlin/DALEX/commit/64ea4b14aa444a5a3e5c3142fb33de80146880b7) | Native Arena resources, EDA and dataset loaders |
| [c219b160](https://github.com/brunocarlin/DALEX/commit/c219b160c818b4405147c735eb7bac2c82570000) | HTTP no-conversion guard |
| [650a11fe](https://github.com/brunocarlin/DALEX/commit/650a11fea692eaa83654d4984e06193ccecd1abe) | Exact aggregate grids/missing states and native surrogates |
| [1ac966b2](https://github.com/brunocarlin/DALEX/commit/1ac966b2a3a1130963000dcf9a1ad5284582e52d) | Native automatic-grid filtering/reductions |
| [783f0697](https://github.com/brunocarlin/DALEX/commit/783f06976b659bfeb7a0ea50839885371a09187f) | Native custom labels, browser-free dendrograms and upstream compatibility CI |
| [080f946b](https://github.com/brunocarlin/DALEX/commit/080f946b4bb3f53134957875d8e7b3fbd854d71c) | Avoid redundant clean-column/grid work and preserve Decimal interventions |

The approach follows the compatibility separation illustrated by
[Plotly's Narwhals migration](https://plotly.com/python/v6-migration/),
[Pandera's backend integration](https://pandera.readthedocs.io/en/latest/narwhals_backend.html)
and [Narwhals' stable API policy](https://narwhals-dev.github.io/narwhals/backcompat/),
with DALEX-specific numerical and integration evidence above.
