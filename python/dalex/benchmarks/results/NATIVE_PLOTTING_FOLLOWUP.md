# Native plotting follow-up

This bounded follow-up closes eager SHAP metadata conversion and moves additional plot preparation off pandas. **It does not complete every remaining pandas path.** The remaining work is listed below.

Implementation commits: [fa4ff30](https://github.com/brunocarlin/DALEX/commit/fa4ff30b499fa6330ae23f431a4d0e0d2594c822) and [cec2f25](https://github.com/brunocarlin/DALEX/commit/cec2f25862315332387c3978fd16519c42ad2473). The final library fingerprint is `8eaed025ecb1b95049bc5e363f385bfa4c80be8031e823935f06e52be5fc5bb5`. Later report-only commits do not change that implementation.

## Changes

- External SHAP keeps observation metadata native and snapshots mutable Polars input. The pandas `new_observation` attribute is lazy and remains writable; native exports and plotting respect explicit edits.
- Native preparation now covers variable importance, residual diagnostics, performance ECDF/ROC/lift, Break Down/interactions and DALEX Shapley plots. Plotting does not publish those legacy tables implicitly. Shapley bars use typed numeric contributions without flattening mixed original feature metadata.
- ROC plotting and scalar AUC share one vector calculation. Boolean/missing targets and exact nullable UInt64 decision scores retain their intended groups and denominators. Lift plots work below 101 rows and across unequal model sizes.
- Compatible numeric plot schemas are promoted explicitly. Unsafe large-integer-to-float combinations raise an actionable error on native paths instead of silently rounding. Pandas-only combinations retain their historical behavior. Arrow string storage is reconciled for mixed pandas 3/native plots.

## Validation

- Full local Python 3.14 suite on the final implementation: **1,322 passed / 18 skipped**. All 18 skips belong to optional legacy PPS tests, which are covered in their compatible CI environment.
- **107 new focused cases** pass on the primary stack, minimum Polars 1/Narwhals 2, pandas nightly, and pandas 3 with Narwhals 2. Tests include real Tree/Kernel SHAP fitting, exports, plots and applicable pickle paths; full-suite existing sklearn/LightGBM integrations also run. Strict guards forbid pandas construction across native fit-to-plot workflows.
- [Implementation CI: 21/21 jobs passed](https://github.com/brunocarlin/DALEX/actions/runs/41149613920). Audited artifacts contain 15 full suites (each 1,322 passed / 18 skipped), eight PyTorch CPU cases, eight TensorFlow CPU cases and 18 PPS cases. Coverage includes Python 3.10–3.14, Windows/Linux/macOS, pandas 3, minimum dependencies and Linux 3.14t.
- **53 complete figure payloads** match pinned, unmodified upstream at rtol=atol=1e-12, including ties, sampling, baselines, hover text and layouts. This comparison now runs in the upstream-compatibility CI job.
- **27 explanation outputs** match upstream at 1e-12; local maximum absolute error is 4.44e-16; CI maximum is 8.88e-16. **15 actual upstream pickle objects** load, with legacy CP/Aspect reuse checked.
- Python 3.14t reports the GIL **disabled after all 21 dependency imports**, then passes the full suite. Without the forcing environment, the isolated import probe still identifies SHAP as enabling the GIL. That remains an upstream dependency limitation; no extension safety declarations are patched.

## Plotting measurements

These are **plotting-only microbenchmarks of identical precomputed explanations**, not fitted-model or end-to-end explanation speedups. Reference: e80dc6a. Candidate: cec2f25. Ordinary Windows Python 3.14.4; pandas 2.3.3, Polars 2.1.0, Arrow 26, Narwhals 2.28.0. BLAS/model threads one; Polars eight. Five trials per case/version run in separate interpreters, with alternating version order and an excluded warm-up. Input/result construction and JSON serialization are excluded. Each timed plot starts with an unread compatibility view. No other DALEX tests or benchmarks run concurrently; desktop activity remains uncontrolled.

Workloads: 500-variable importance; 100,000-row diagnostics sampled to 1,000 points for LOWESS; 100,000-row ROC; and 500-variable/10-path Shapley. RSS is sampled every 2 ms. Allocator caching affects baseline RSS, and short operations can peak between samples. The raw data include every trial, quartiles, absolute peak and baseline RSS, source fingerprints and figure hashes.

| Plot | Backend | Previous median ms | New median ms | Previous/new time | RSS increase MiB, previous → new | Peak RSS MiB, previous → new |
|---|---|---:|---:|---:|---:|---:|
| importance | pandas | 18.04 | 18.95 | 0.952 | 0.08 → 0.07 | 190.00 → 189.92 |
| importance | polars | 18.22 | 17.96 | 1.014 | 0.18 → 0.35 | 192.60 → 196.40 |
| importance | pyarrow | 17.65 | 16.59 | 1.064 | 0.09 → 0.15 | 191.38 → 198.09 |
| diagnostics | pandas | 92.04 | 93.08 | 0.989 | 19.41 → 11.43 | 287.77 → 279.11 |
| diagnostics | polars | 98.77 | 89.47 | 1.104 | 11.00 → 0.96 | 283.76 → 262.28 |
| diagnostics | pyarrow | 98.97 | 89.69 | 1.104 | 10.01 → 0.84 | 280.43 → 258.79 |
| performance | pandas | 33.64 | 22.67 | 1.484 | 6.53 → 3.52 | 203.14 → 199.50 |
| performance | polars | 38.38 | 21.52 | 1.783 | 11.48 → 0.07 | 211.00 → 194.98 |
| performance | pyarrow | 36.86 | 21.72 | 1.697 | 8.73 → 0.00 | 206.52 → 195.80 |
| shap | pandas | 20.13 | 17.40 | 1.157 | 0.38 → 0.05 | 191.11 → 190.43 |
| shap | polars | 26.83 | 17.74 | 1.513 | 0.85 → 0.14 | 195.48 → 194.40 |
| shap | pyarrow | 25.05 | 16.58 | 1.510 | 0.65 → 0.08 | 193.07 → 191.43 |

Ratios above one favor this change; ratios below one are regressions. All 60 before/after figure comparisons pass at 1e-12. The initial fa4ff30 measurements are retained separately: they exposed extra Arrow ROC allocation before the shared vector kernel. Do not attribute phase-to-phase timing differences solely to that refactor; desktop/source-order effects are uncontrolled.

Earlier million-row sklearn/LightGBM explanation benchmarks still belong to implementation 080f946b, not to these plotting measurements. Their pandas partial/conditional computation and importance regressions, and wider-frame Polars memory overhead, have not been erased or remeasured away by this work. The current CI also reruns six cold/warm real-model importance reports; all numerical parity errors are zero.

## Remaining work

1. Migrate CP/aggregated-profile, Aspect/triplot and fairness plot preparation. Preserve typed partitions, group display labels, observation indexes, edited pandas views and native precision.
2. Consolidate older private pandas explanation algorithms into adapters over native kernels. ROC/AUC is consolidated here, but the older Break Down, Shapley and aggregate reference helpers remain.
3. Extend strict fit-to-plot conversion guards through the remaining families and external integrations.
4. Profile and address the separately reported explanation-time and memory regressions. Native storage alone does not guarantee faster pandas or Polars execution.

Writable pandas APIs and NumPy/SciPy/tensor boundaries are intentional. Pandas-free packaging, lazy/distributed/GPU engines, multiclass and arbitrary object-column/concurrency cases are separate uncertified scope. KernelExplainer retains an upstream local closure and is not claimed to support ordinary pickle. Python >=3.10 remains a proposed compatibility change for maintainer review.

PR [#587](https://github.com/ModelOriented/DALEX/pull/587) remains **draft and unmerged**. The upstream default branch is unchanged. Fork CI does not grant approval to run the upstream PR workflow.

Raw evidence: [final plotting trials](native-plotting-followup/plot-benchmark-final.json), [initial plotting trials](native-plotting-followup/plot-benchmark-initial.json), [CI audit](native-plotting-followup/ci-validation.json), [local test counts](native-plotting-followup/local-validation.json), [artifact fingerprints](native-plotting-followup/manifest.json).
