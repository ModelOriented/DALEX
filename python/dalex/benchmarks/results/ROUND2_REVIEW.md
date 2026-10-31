# Second refinement: readiness for review

The implementation is ready for maintainer review within the documented scope,
subject to the author's manual review. PR [#587](https://github.com/ModelOriented/DALEX/pull/587)
remains draft and unmerged. This is not a claim of universal backend compatibility,
compiled-extension thread safety, or faster inference.

## Reviewed source and reproducible evidence

Source [03545055](https://github.com/brunocarlin/DALEX/commit/035450550ee922f07a3155cc1675ef6ac73ce9c9)
passes [all 20 fork checks](https://github.com/brunocarlin/DALEX/actions/runs/40804809506).
The accompanying evidence commit changes documentation and measurement records;
the exact tested library tree and normalized 117-module fingerprint are in
[round2-validation.json](round2-validation.json).

- Fifteen full-suite jobs each report **562 passed, 18 skipped, zero failures/errors**.
  The matrix covers Python 3.10–3.14, Linux/macOS, Windows 3.14, minimum pandas,
  minimum Polars/Narwhals, pandas 3 and Linux 3.14t. No job permits failure.
- The 18 legacy PPS cases run separately: **18 passed, zero skipped**. Required
  PyTorch and TensorFlow CPU integration jobs each pass **eight tests**.
- Final local Windows source, installed wheel, Polars 1.0/Narwhals 2.0, and
  pandas 3.2.0.dev0+455 suites each pass **562/18**. The wheel is imported from
  site-packages outside the checkout; all 117 Python modules match the source.
  Wheel/sdist hashes and version records are included. Primary installation was
  restored to editable source and dependency checks pass.
- **23 legacy numerical outputs match unmodified upstream 77ea3ad9 with maximum
  absolute error zero**. The reference uses only the documented NumPy spelling
  shim. Cross-backend measured benchmark errors are also zero.
- The 3.14t check verifies the actual GIL state after 21 imports. SHAP enables the
  GIL automatically without `PYTHON_GIL=0`; the forced-disabled matrix passes.
  No extension safety declarations are patched. These checks do not certify
  arbitrary concurrent calls into compiled dependencies. Neural checks use
  ordinary CPU interpreters, not free-threaded neural certification.

## What this refinement fixes

Native observation/replacement paths now distinguish Arrow/Polars nulls from
valid NaNs and preserve untouched nullable UInt64/Int64 values, including values
outside exact float64 range. Observation validity is cached without leaking
helper columns into public result tables. Tests independently exercise breakdown,
Shapley, Aspect, conditional profiles, and explicit/all-missing profile grids.
Mixed missing intervention bins that cannot be represented by the existing
pandas aggregate key raise a clear error; raw profiles retain both states.

Arena obtains native row snapshots instead of coercing whole feature tables.
Its static/HTTP JSON boundary maps missing and non-finite presentation values to
JSON null without changing model inputs or caches. Reused custom callbacks remain
idempotent and survive serialization. Grouped profiles retain missing groups and
omit unused categorical levels, with independently computed expected results.

Arrow's pandas metadata no longer silently hides a physical feature column or
forces fractional interventions back to stale integer dtypes. Custom schema and
field metadata remain intact. Legacy pandas Arrow-extension correlation uses an
explicit float statistics copy while native model inputs retain exact values.
Redundant native copies are removed; failure-path tests verify caller inputs are
unchanged. Thirty-two Polars-output sklearn/LightGBM pipeline cases include
portable selectors and the explicit model-owned bridge needed by sklearn <1.9
with Polars 2. The bridge was also checked locally on sklearn 1.7 and 1.8.

## Performance, including regressions

See [complete benchmark tables and protocols](ROUND2_BENCHMARKS.md), with all raw
trials, source/model fingerprints, quartiles and memory samples in [round2/](round2/).
On one million original Polars rows, sampling 1,000, five trials/backend, total
pandas/Polars ratios are **1.053–1.152 for sklearn** and **1.066–1.268 for
LightGBM**. Total includes pandas conversion; construction, training and worker
startup are excluded. Model/BLAS threads are one, Polars eight; total core budgets
are not equal. Peak RSS is approximately **460–484 MiB for pandas versus
385–407 MiB for Polars**. All measured numerical errors are zero.

Several explanation-only stages regress: sklearn breakdown is about **8.3%
slower** with Polars, with partial/conditional profiles about 6–7% slower.
Latest sampled-workflow ratios are smaller than the earlier report's 1.26–2.02;
the reason for between-session timing changes is not established. Wide-frame
ratios remain 3.74/3.58 for sklearn breakdown/Shapley and 1.53/1.54 for LightGBM
at their explicitly identified source snapshot. Wide Polars RSS is higher.
Small before/after improvements cannot establish a causal optimization effect.

## Remaining review and compatibility decisions

- Python >=3.10 drops 3.9 and needs maintainer approval before release. Narwhals
  >=2,<3 and pandas >=1.5.3,<4 are declared. No pandas 4 release was available
  when checked; pandas 3.2 nightly is evidence, not pandas 4 certification.
- Result/statistical/presentation tables intentionally retain pandas. External
  matrix/tensor boundaries can coalesce null/NaN or round large integers; those
  boundaries are not certified for models sensitive to such distinctions.
- LightGBM's tested raw dictionary/Enum and pandas ArrowDtype paths require
  explicit stable training-time encoding. See [native recipes](../../NATIVE_BACKEND_RECIPES.md).
- A legacy Windows Python 3.10/pandas 1.5 LightGBM import-order crash reproduces
  without DALEX. Current Windows 3.14 probes pass; this remains a separate
  dependency issue. The locked conda-forge 3.14t Polars route avoids missing
  PyPI runtime wheels; SHAP's automatic GIL fallback is an upstream limitation.
- Lazy/distributed/GPU inputs, arbitrary estimators/dtypes, every nested/temporal
  intervention, multiclass and complex multi-input neural models remain
  uncertified. Conditional profiles require observed support. Existing
  dependency deprecation warnings remain visible.

Earlier failed attempts exposed real old-Narwhals scalar truthiness, pandas 3
missing-label, Arrow metadata and old sklearn output issues; each has regression
coverage and the final matrix is green. A superseded macOS dependency download
timed out before tests; a local nightly run encountered a Tcl/Tk environment
failure and passed fully with the same headless backend used in CI. No test
timeouts were raised and no new skips were added to obtain the final result.

Upstream/master was fetched and remains 77ea3ad9580a3b68aaa94d8dca75d11bd527848f.
Fork CI does not imply upstream acceptance. AI assistance is disclosed in the PR;
the author's manual review is pending before requesting maintainer review.
