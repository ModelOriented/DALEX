# Fairness plotting follow-up: validated implementation 4fcdda8

[Implementation](https://github.com/brunocarlin/DALEX/commit/4fcdda8ec2dbc1976662244c500909bfcd4371b6) prepares all eight classification fairness plots and both regression plots with native tables. Explicit subgroup positions preserve ordering. Edited pandas result tables remain authoritative. Cutoff plots retain the caller's cutoff mapping; incomplete score matrices align by model and metric instead of silently shifting cells. Legacy private table helpers now adapt the same native kernels and retain their pandas return contracts.

## Validation

- 119 added cases pass, including 16 fitted scikit-learn/LightGBM pipelines, native construction guards, cached-table edits, missing scores, invalid grids and private adapter contracts.
- Exact implementation local full suite: **1,631 passed / 18 skipped**, zero failures. Skips are optional PPS cases, exercised separately in CI.
- [All 21 CI jobs passed](https://github.com/brunocarlin/DALEX/actions/runs/41161343899). Downloaded artifact audit verifies 15 complete suites (1,631/18 each), eight PyTorch cases, eight TensorFlow cases and 18 PPS cases. Matrix includes Python 3.10–3.14, Windows/Linux/macOS, pandas 3, minimum dependencies and free-threaded 3.14t.
- 152 complete figure payloads match pinned unmodified upstream at rtol=atol=1e-12. Twenty-seven explanation outputs match with maximum absolute error 8.88e-16, and 15 actual upstream pickle objects load.
- Actual GIL state is disabled after all 21 dependency imports in the forced 3.14t run. The unforced probe identifies SHAP as enabling the GIL. This dependency limitation remains separate from DALEX failures; arbitrary extension concurrency safety is not certified.
- All current benchmark trials and CI artifacts have the same fingerprint: 134 modules, SHA256 `76d785653779dd92bf9939d02d28e32cb49e7db63cbd187b0483af7c53d34695`.

## Plotting-only benchmark

Identical precomputed fairness objects: two classification models, 1,024 fixed prediction rows each, four equally sized subgroups, identical cutoff and epsilon. Construction and score computation are outside the timer. Reference 8ab8ac1; candidate 4fcdda8. Windows Python 3.14.4, pandas 2.3.3, Polars 2.1.0, Arrow 26, Narwhals 2.28.0. Polars uses eight threads; BLAS/model pools one.

Five isolated trials per backend/version alternate source order and exclude warm-up. RSS sampling interval is 2 ms. No other DALEX tests or benchmarks ran concurrently; desktop activity, allocation caching and short sampling gaps limit interpretation.

| Backend | Previous/new median ms | Additional RSS MiB, previous → new | Peak RSS MiB, previous → new |
|---|---:|---:|---:|
| pandas | 104.92 / 111.25 | 0.133 → 0.375 | 200.37 → 200.46 |
| Polars | 110.19 / 103.23 | 0.371 → 0.496 | 206.19 → 207.03 |
| PyArrow | 92.68 / 86.63 | 0.406 → 0.367 | 202.22 → 201.84 |

All 15 before/after figure pairs pass numerical/presentation comparison at 1e-12. Polars and Arrow medians improved about 6–7% in this case, while **pandas regressed about 6%**. Additional RSS increased for pandas and Polars. These small plotting-only changes do not establish faster fairness calculations or end-to-end model workflows. Earlier pandas PDP/conditional computation and importance regressions, and wide-frame Polars memory overhead, remain unresolved at this checkpoint.

## Remaining scope at this checkpoint

Aspect/triplot plot preparation and consolidation of older private pandas algorithms remain. Public writable pandas views, index/scalar formatting and external NumPy/SciPy/tensor contracts are deliberate boundaries. Lazy/distributed/GPU engines, multiclass and arbitrary object/concurrency support are uncertified. Python >=3.10 is proposed for maintainer review. Published pandas 3.1.0 and the tested 1.5/2.x/3.x stacks must not be described as pandas 4 validation.

PR [#587](https://github.com/ModelOriented/DALEX/pull/587) remains draft and unmerged. [Raw measurements](native-fairness-plotting-followup/plot-benchmark.json), [CI audit](native-fairness-plotting-followup/ci-validation.json), [artifact fingerprints](native-fairness-plotting-followup/manifest.json).
