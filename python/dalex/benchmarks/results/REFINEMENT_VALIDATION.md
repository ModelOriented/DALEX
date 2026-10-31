# Narwhals migration: refinement review evidence

Historical report. See [second refinement review](ROUND2_REVIEW.md) for the latest
562-test source validation and explicitly source-stamped benchmarks.

## Browser-free completion follow-up

Repeated CI runs exposed intermittent Chrome startup/shutdown hangs even after
the earlier three-to-one render reduction. Triplot construction now uses its
existing numeric coordinates to choose hover-point density, without Kaleido or
a headless browser. Existing line vertices, endpoints and node locations are
retained; the number of intermediate hover points may change. Constant panels
use a positive fallback span. Twelve pandas/Polars/Arrow widget cases prevent
external rendering and check vertex/end-point preservation, including constants.

The complete source and rebuilt installed-wheel suites now each pass **360 / 18**.
The retained 18 skips still execute separately in the compatible PPS job. See
[browserless-validation.json](browserless-validation.json) for the tested library
content hashes, exact wheel/sdist hashes and updated counts. The latest CI
status is linked from the [draft PR](https://github.com/ModelOriented/DALEX/pull/587).
The historical 354/18 and 353/18 snapshots below remain explicitly identified.

## Arrow metadata audit follow-up

[b7f7ef3c](https://github.com/brunocarlin/DALEX/commit/b7f7ef3cde3e8bd19d2dcad070a45174cf9b58ae)
adds a strict metadata-sensitive model callback regression test and restores the
complete Arrow schema after feature permutations, using the existing immutable
column buffers. Narwhals' name-based replacement had dropped replaced-field
metadata despite preserving values/types. The suite now has **354 passed / 18
PPS skips**; the source and rebuilt installed-wheel suites both pass. Latest
checks are available on the [draft PR](https://github.com/ModelOriented/DALEX/pull/587).
Exact follow-up results and distribution hashes are in
[arrow-metadata-validation.json](arrow-metadata-validation.json).

The results below are the retained pre-follow-up snapshot, explicitly stamped
with its tested commits and **353/18** counts. The Arrow schema restoration does
not change pandas/Polars values or processing; benchmark measurements retain their
original source stamps rather than being reassigned to the follow-up.

## Source-stamped validation baseline

Library implementation: [42fd07b2](https://github.com/brunocarlin/DALEX/commit/42fd07b2bbf2a8418ea2efad8a1f3e1e5970a856).
Benchmark instrumentation: [4d471716](https://github.com/brunocarlin/DALEX/commit/4d4717166e400287220560160b2c7c038339e6ab).
This report supersedes earlier validation counts; historical measurements retain
their original source stamps. Exact job links, JUnit totals, versions and
distribution hashes are in [refinement-validation.json](refinement-validation.json).

## Checks and numerical parity

[All 20 fork CI jobs pass](https://github.com/brunocarlin/DALEX/actions/runs/40626293223).
Fifteen complete-suite jobs each report **353 passed, 18 skipped, zero failures
or errors**. They cover Python 3.10–3.14 on Linux/macOS, Windows 3.14, pandas 3,
Linux 3.14t with the GIL disabled, pandas 1.5.3/Narwhals 2.0, and Polars 1.0/Narwhals
2.0. The three benchmark jobs check NumPy-bound sklearn, native sklearn and native
LightGBM paths. No job allows failure.

The 18 main-suite skips are the inherited Aspect/PPS cases. They now run,
**18 passed / zero skips**, in the compatible pandas 1.5.3/scikit-learn <1.7/PPS
1.3 environment. The four legacy Arena tests previously guarded by an unreachable
`sys.platform.startswith("ubuntu")` condition now run in the ordinary matrix.
Required CPU neural jobs each pass **eight** tests: PyTorch 2.14.1+cpu/Python 3.14
and TensorFlow 2.21/Python 3.13, with pandas/Polars/Arrow, Deep/Gradient SHAP,
trainable/frozen models, plotting, additivity and unchanged-weight checks.

Local Windows complete suites each pass **353/18** on ordinary 3.14.4,
disabled-GIL 3.14.10, pandas **3.2.0.dev0+442.gcd83fbf6fd**, and Polars 1.0/Narwhals
2.0. Those four runs precede the subsequent layout-only renderer optimization;
the exact optimized source is covered by CI's full suites, local PPS **18/0**,
and six renderer integration cases on ordinary, disabled-GIL and nightly Python.
Minimum-dependency focused checks pass **124** cases. Shared sklearn/LightGBM
prediction on Polars **2.0** also passes eight threaded cases, including N=None.

The rebuilt wheel and source archive build successfully. A separate environment
imports the wheel from **site-packages outside the checkout** and passes the
entire **353/18** suite with the renderer optimization. Eight installed-wheel
PyTorch cases and eight source TensorFlow cases passed before that layout-only
change; neural library code is unchanged. Setuptools configuration/data-discovery
warnings remain visible, and packaged-dataset tests pass.

**23 numerical outputs match unmodified upstream
77ea3ad9580a3b68aaa94d8dca75d11bd527848f with maximum absolute error 0**. Coverage
includes grouped weighted importance/raw permutations, breakdown/interactions,
Shapley values, profiles, performance/diagnostics, five Aspect/triplot outputs
and sampled approximate Kernel SHAP. The reference process aliases the removed
one-dimensional `np.in1d` spelling to `np.isin` on NumPy 2; upstream files are
unchanged. [Per-output errors](refinement/legacy-numerical-parity.json).

## Refinements and intentional corrections

- Production imports use `narwhals.stable.v2`, tested at 2.0 and current versions.
  [Narwhals' stability guidance](https://narwhals-dev.github.io/narwhals/backcompat/)
  applies to its API; backend dtype/extension behavior still needs testing.
- Sampling/permutation validates integer positions before narrowing, including
  oversized unsigned integers. Repeated/negative/empty gathers preserve order
  and schemas. Native observation/matrix reconstruction preserves tested nullable,
  timezone/nanosecond, duration, nested, Decimal and Arrow metadata cases.
- Arrow dictionary nulls are decoded at NumPy boundaries, so masked physical
  dictionary values cannot leak into categories. Bare Arrow arrays are wrapped
  as one existing chunk for Narwhals' vector boundary. No input buffers are copied
  for that wrapper.
- Numeric grids ignore missing observations; all-missing numeric columns have one
  missing split. Partial profiles retain missing interventions; conditional
  profiles match them to missing originals. Numeric accumulated-effect grids with
  missing points raise an explicit error because those points have no numeric
  ordering. Missing numeric bins are retained in result tables; categorical
  profiles are appropriate when a missing level must have a visible axis label.
- Conditional categorical profiles match original values before formatting.
  This corrects pre-existing pandas zero-weight results for numeric/boolean codes
  and distinguishes missing values from literal display strings. These missing/
  categorical corrections intentionally change previously broken pandas outputs;
  independent expected-value tests supplement backend parity.
- Fairness accepts native protected vectors and one-column frames; mitigation also
  normalizes native targets. Regression fairness uses normalized string memberships
  consistently. The caller defines category encodings and missing-group policy.
- Wide single-column interventions inspect only changed columns, rather than the
  complete schema each time. Already-owned pandas work frames avoid redundant
  deep copies. Integer-population sampling preserves MT19937 draws without
  allocating an identity population. Core Shapley observation metadata converts
  once per explanation, while model batches retain their native backend.
- Old Polars export borrows are isolated by buffer-sharing clones on Polars 0/1.
  Polars 2 uses internal read/write locks; source inspection, standalone export
  stress tests and shared-model tests support avoiding that wide-frame clone cost
  on the tested 2.0/2.1 releases. Arbitrary models/options are not certified safe.
- Plotly 6+ widgets declare their required `anywidget` dependency. Triplots resolve
  all axis ranges from one rendered figure instead of three. The restored PPS CI
  initially hit a 120-second Fifa triplot timeout; the single-render fix passes
  all 18 cases without removing tests or increasing that timeout. Widget integration
  checks assert one actual render and construct the resulting HBox.

## Performance and memory

[Tables, protocols, IQRs and raw measurements](REFINEMENT_BENCHMARKS.md) retain
improvements **and regressions**. Windows desktop measurements use identical
models/data/seeds/settings within backend pairs. Model/BLAS/OpenMP threads are one;
Polars/Arrow retain engine defaults unless a pool variant is stated. Training and
process startup are excluded, conversions are included where specified, workers
are isolated and backend order alternates.

| Workload | sklearn pandas/Polars ratio | LightGBM pandas/Polars ratio |
|---|---:|---:|
| 128 rows, 1,000 features: breakdown | 3.585 | 1.479 |
| 128 rows, 1,000 features: DALEX Shapley | 3.691 | 1.475 |
| One million rows, all rows sampled: importance | 1.034 | 1.068 |
| One million original Polars rows, sample 1,000: six workflows | 1.262–1.822 | 1.260–2.016 |

Ratios above one favor Polars. Dataframe-only million-row permutation ratios are
5.93/6.11, but inference dominates end-to-end time. The sampled Polars-origin
workflow gains include avoiding a full pandas conversion; some explanation-only
phases are still approximately **5% slower**. Compared with unmodified upstream
pandas on the same saved wide sklearn model, migration pandas medians are **0.5%
and 1.1% slower** for breakdown/Shapley. Three trials do not establish statistical
significance for those small differences.

Million-row importance peak RSS is **705.8/642.9 MiB** (pandas/Polars) for sklearn
and **696.2/544.9 MiB** for LightGBM. Polars uses roughly **6–7 MiB more** in the
small-row/wide-frame workload. Sampling RSS every 2 ms can miss peaks; allocator
warming and ordinary desktop/OS variation affect measurements. One-thread Polars
sensitivity results do not justify a blanket global override: pool variants ran
in fixed order, and Arrow conversion retains its own defaults. No universal
acceleration or free-threading speedup is claimed.

## Runtime and dependency boundaries

Python **>=3.10** is proposed; dropping 3.9 needs maintainer agreement. Narwhals is
>=2,<3 and pandas >=1.5.3,<4. Python 3.14 is primary. PyPI checked on 2026-10-28
has pandas 3.1.0 and no 4.x release; the tested nightly is 3.2. No pandas 4
compatibility is claimed, and the major-version cap remains precautionary.

Python 3.14t uses the committed conda-forge interpreter/Polars lock, with scientific
PyPI dependencies. The actual GIL is disabled after imports, not inferred from
the executable name. The archived CI snapshot checks 17 imports; the expanded
guard additionally checks Flask-CORS, requests, IPython and Kaleido (**21 total**)
and passes locally, with its source hashes/probe recorded. **SHAP's compiled
extension enables the GIL without PYTHON_GIL=0.** Forced-disabled tests do not
certify arbitrary concurrent compiled SHAP calls; no extension flags are patched.
Neural jobs use ordinary CPU Python. Stable TensorFlow lacks a cp314 wheel.

A separate old Windows Python 3.10/pandas 1.5 LightGBM stack can crash during
standalone training when pandas imports first: three NumPy/pandas/Polars cases
fail without importing DALEX. Importing LightGBM first passes all three; the
current ordinary 3.14 stack passes all six. This is recorded as an upstream
dependency/import-order limitation, rather than a DALEX numerical regression.
The current Windows 3.14 and minimum Linux CI suites pass.

Initial local concurrent widget/PPS attempts also encountered a full C: volume
and Chrome startup failures. Cache cleanup and retiring disposable validation
environments restored space; complete suites, PPS and installed-wheel validation
subsequently pass. Those failed attempts are not used as timing comparisons.
Only the primary ordinary Python environment is intended to remain installed;
the CI manifests/requirements and recorded versions reproduce retired environments.

Native pandas/Polars/Arrow features cover the tested explanation families, LIME,
SHAP boundaries, surrogates, Aspect and Arena's 13 plot families. Result/statistical
tables and presentation metadata intentionally remain pandas-backed. Categorical
LightGBM inputs require a stable explicit encoding: its Arrow-stream path rejects
raw dictionary/Enum columns, and the tested LightGBM rejects pandas ArrowDtype
features. Older sklearn pipelines need portable positional selectors/explicit
numeric transformations. See [recipes](../../NATIVE_BACKEND_RECIPES.md).

Lazy/distributed/GPU backends, arbitrary estimator/dtype policies, every nested/
temporal counterfactual, complex/multi-input neural networks and all possible
third-party concurrent calls remain outside this evidence. Warnings include
pandas' future convert_floating removal, legacy chained-assignment warnings,
SHAP NumPy/RNG deprecations and Keras input-structure warnings.

The [upstream PR](https://github.com/ModelOriented/DALEX/pull/587) remains **draft**
and unmerged. Upstream/master remains at the reference commit after fetching.
Fork CI evidence does not imply upstream maintainer approval or upstream CI success.
The implementation was prepared with AI assistance; the author's manual review is
pending before requesting maintainer review.
