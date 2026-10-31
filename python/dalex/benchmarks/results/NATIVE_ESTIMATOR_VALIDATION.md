# Native estimator migration: final review evidence

> Historical snapshot. See [refinement validation](REFINEMENT_VALIDATION.md) for current coverage.

Tested implementation: `08e8c380f2fb2212fe0620206d8bb61c9819f780`.
The following evidence/documentation commit does not change library code.
This report supersedes the earlier validation snapshots for current coverage.

## Checks

[All 18 required CI jobs passed](https://github.com/brunocarlin/DALEX/actions/runs/40604998136).
Each of the 12 main-suite jobs reports **188 passed, 22 inherited skips, zero
failures/errors**: Python 3.10–3.14 on Linux/macOS with pandas 2, an additional
3.14/pandas 3 job, and Linux 3.14t with the GIL disabled. Neither job failures nor
new backend tests are waived. The minimum-pandas/Narwhals/PPS job also passed.

Two required CPU neural jobs each pass four integration tests: PyTorch
2.14.1+cpu on Python 3.14, and TensorFlow 2.21 on Python 3.13. They exercise
Deep/Gradient SHAP through predict_parts/model_parts and plots, across pandas,
Polars and Arrow, with an independent linear-model additivity check. Stable
TensorFlow has no cp314 wheel, so that integration uses 3.13 rather than claiming
3.14 or free-threaded TensorFlow support. Complex/multi-input/GPU networks remain
outside this evidence.

Local Windows full-suite checks also pass **188/22** on ordinary 3.14.4,
free-threaded 3.14.10 and the pandas nightly. These test suites were run
concurrently; their elapsed times are validation evidence, not performance
comparisons. A fresh environment installs the built wheel and runs tests copied
outside the checkout: **128 migration tests plus four PyTorch integration tests
pass**, with imports asserted to resolve to site-packages. Both wheel/sdist build;
hashes, timings and per-job JUnit totals are in `native-estimator-validation.json`.

The 22 inherited skips remain visible: 18 Aspect/PPS tests and four old Arena
tests guarded by an Ubuntu platform string. New integration tests exercise those
families independently; optional PPS is checked in its compatible legacy environment.
Warnings include dependency deprecations and Keras/SHAP input-structure warnings;
the tests do not suppress them or treat them as compatibility guarantees.

## Models and native features

- The same fitted scikit-learn HistGradientBoosting and LightGBM 4.7 models train
  on Polars and explain pandas/Polars/Arrow inputs. Regression and binary
  classification cover importance, breakdown, DALEX Shapley values, profiles,
  performance and external Tree SHAP. Call guards ensure native feature batches
  reach prediction; estimator-specific SHAP uses explicit matrix/tensor boundaries.
- A RandomForest/ColumnTransformer pipeline handles numbers, categories and
  timezone-aware date preprocessing without converting the original feature frame
  to pandas. Older sklearn/Polars 2 stacks need positional selectors and an explicit
  numeric FunctionTransformer: older named-selection/passthrough paths expect
  pandas indexing metadata before DALEX is even constructed. The portable recipe
  passes on sklearn 1.6.1/pandas 1.5/Narwhals 2 as well as current versions.
- Arrow column labels now use names rather than Arrow's column-array objects.
  Arrow also joins LIME, kernel SHAP, surrogate and Aspect/triplot numerical
  comparisons. Polars and pandas retain their existing label/index semantics.
- Timezone-aware datetime, Decimal, list and struct fields preserve their schemas
  and original data through gathers/permutation importance. Nullable categorical,
  string/integer and all-null numeric metadata cases are covered. This does not
  certify every counterfactual/statistical operation on nested or temporal fields.
- LightGBM's native Arrow stream rejects dictionary-encoded categorical columns
  (observed as `Unsupported Arrow type: dictionary`). A tested explicit training
  encoding handles categories and missing values. DALEX does not invent category
  codes that might disagree with training; use consistent encoding/preprocessing.
- PyTorch SHAP receives tensors matching model dtype/device. GradientExplainer's
  sampled tensors are cast at a model adapter boundary without changing weights or
  process-wide default dtype. Scalar expected values/output dimensions are
  normalized for plots, including TensorFlow EagerTensor baselines.

## Numerical compatibility and concurrency

The unmodified upstream comparison now checks **23 numerical outputs**, all with
maximum absolute error **0**. It includes the original core methods, five Aspect/
triplot outputs, and an approximate nonlinear Kernel SHAP case with sampled
background rows. The only reference compatibility shim aliases removed np.in1d
to np.isin for the reference's one-dimensional Aspect checks; upstream files are
unchanged. See `native-estimator-legacy-parity.json` and `ci/compare_legacy.py`.

Explicitly seeded core explanations use local MT19937 streams. Tests run four
concurrent calls for importance, Shapley values, profiles and Aspect default/Shap
methods and compare against serial references while checking global RNG state.
Shared scikit-learn and LightGBM models also have seeded threaded-importance tests.
Calls without an explicit seed retain legacy global-RNG semantics.

External Kernel/Gradient SHAP use global NumPy internally. Cooperating wrapper
calls use a scoped lock/state restore; sampled Kernel SHAP resumes the same RNG
state as legacy pandas after background sampling. Approximate backend comparisons,
repeated calls and four concurrent calls on one shared Polars explainer pass on
ordinary and free-threaded Python. Arbitrary external NumPy consumers, stochastic
user models and unseeded concurrent calls are not covered by that guarantee.

The expanded tests exposed an earlier macOS stall in threaded Polars Aspect
calls. The revised column-wise diagnostic avoids full mixed-frame NumPy
materialization, also fixing nested-column constructor failures. Final macOS
jobs pass the concurrency tests; CI uses a watchdog timeout that can terminate
blocked worker threads instead of waiting indefinitely in teardown.

## Free threading and pandas versions

The conda-forge interpreter/Polars lock plus PyPI scientific wheels resolves the
missing cp314t Polars wheel route. The required job verifies Py_GIL_DISABLED and
the actual GIL state after every dependency import, including LightGBM, before
running the full suite. Runtime/probe/version files are archived with this report.

SHAP's shap._cext still automatically enables the GIL when the override is absent.
Passing tests with PYTHON_GIL=0 does not certify arbitrary simultaneous compiled
SHAP calls as safe. Neural integrations are tested on ordinary CPU Python, not
3.14t. No extension safety flags are patched and no free-threading speedup is claimed.

No pandas 4 release/prerelease/nightly was available on 2026-10-27. The available
nightly **3.2.0.dev0+442.gcd83fbf6fd** passes the entire 188/22 suite; exact revision
and versions are in `pandas-nightly-validation.json`. The pandas<4 cap is a
precautionary major-version policy for maintainer review, not a demonstrated
pandas 4 failure or a claim that pandas 4 was tested.

## Performance and remaining scope

[Native estimator benchmark tables](NATIVE_ESTIMATOR_BENCHMARKS.md) distinguish
two protocols: native-model permutation importance with all rows sampled up to
100k, and six explanation families starting from a 100k-row Polars frame with a
1k explanation sample. The latter includes the pandas input conversion and also
reports explanation-only time separately. Every paired comparison uses one fitted
model, identical values/weights/seeds/settings and isolated, alternated trials.
All measured numerical parity errors are 0.

Warm 100k-row permutation medians are pandas/Polars **2.8325/2.7764s** for sklearn
and **2.5441/2.4170s** for LightGBM. Dataframe-only ratios are 6.620/7.056, but
inference dominates end-to-end time. Polars RSS is often higher at small sizes;
in that LightGBM 100k trial it is lower (267.7/264.7 MiB pandas/Polars).
Both improvements and regressions are retained. Several explanation-only cases
remain slower with Polars, even where the original-Polars workflow saves input
conversion time/memory. Before/after raw snapshots and all latest CI cold/warm
model variants are committed; timing variance does not establish universal gains.
RSS sampling every 2ms can miss peaks and warm allocator state affects baselines.

Pandas remains the compatible public result/statistics/plotting dependency.
Lazy frames are rejected rather than implicitly collected; distributed/GPU/other
Narwhals backends require their own environments and contracts. Certified scope
is the models/methods/dtypes/platforms tested above, not every possible model or
dataframe type. Upstream/master remains 77ea3ad9580a3b68aaa94d8dca75d11bd527848f.
PR #587 remains draft for the author's manual review; upstream Actions require
maintainer approval and no upstream branch has been merged or changed.
