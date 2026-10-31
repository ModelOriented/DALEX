# CP plotting follow-up: validated implementation 8ab8ac1

[Implementation](https://github.com/brunocarlin/DALEX/commit/8ab8ac12c5fe661ce0fadcee38176759f978e2ef) moves CP line/bar preparation and aggregate raw-profile overlays to native tables. Observation indexes align original predictions before models are combined. Typed axes and marker comparisons remain separate from scalar hover formatting; exact UInt64 and nanosecond values retain their precision. Edited pandas result/observation views are authoritative. Pandas publication uses a shared column dtype formatter without the deprecated convert_floating flag.

## Validation

- 91 focused cases pass on the primary stack, minimum Polars 1/Narwhals 2, and pandas nightly. Coverage includes default and edited metadata, multiple models/schemas, exact integer markers, dates/timezones/durations/Decimals, public pandas dtypes and native construction guards.
- Exact implementation local full suite: **1,512 passed / 18 skipped**, zero failures. All 18 skips are optional PPS cases; CI tests those separately in their compatible environment.
- [All 21 implementation CI jobs passed](https://github.com/brunocarlin/DALEX/actions/runs/41158454650). Artifact audit: 15 complete suites (1,512/18 each), eight PyTorch CPU cases, eight TensorFlow CPU cases and 18 PPS cases. Python 3.10–3.14, Windows/Linux/macOS, pandas 3, minimum dependencies and 3.14t are covered.
- 132 complete figure payloads match pinned, unmodified upstream at rtol=atol=1e-12, including 49 CP cases. Twenty-seven explanation outputs match (CI maximum absolute error 8.88e-16), and 15 actual upstream pickle objects load.
- 3.14t reports the GIL disabled after all 21 imported dependencies and passes the complete suite. Unforced SHAP import still enables the GIL; that dependency limitation and arbitrary extension concurrency safety remain separate.

## Plotting benchmarks

These are **plotting-only** measurements of identical precomputed explanations, excluding construction, model fitting and prediction. They do not establish faster CP explanation calculations or end-to-end model workflows. Reference: 4818d9f; candidate: 8ab8ac1. Windows Python 3.14.4, pandas 2.3.3, Polars 2.1.0, Arrow 26, Narwhals 2.28.0. Model/BLAS threads one; Polars eight. Twenty observations, eight variables and 31 grid points produce 4,960 raw profile rows; overlay cases also contain 248 aggregated rows.

Five isolated trials per backend/version alternate source order and exclude a warm-up. Each trial starts with unread compatibility views; current plots publish none. RSS is sampled every 2 ms. No other DALEX tests or benchmarks ran concurrently. Desktop activity, allocator caching, short sampling gaps and different library thread counts limit generalization.

| Plot | Backend | Previous/new median ms | Previous/new time | RSS increase MiB, previous → new | Peak RSS MiB, previous → new |
|---|---|---:|---:|---:|---:|
| ceteris-paribus | pandas | 505.52 / 425.17 | 1.189 | 16.68 → 15.57 | 217.36 → 216.54 |
| ceteris-paribus | polars | 500.72 / 340.00 | 1.473 | 15.52 → 13.14 | 221.62 → 227.42 |
| ceteris-paribus | pyarrow | 491.26 / 347.84 | 1.412 | 16.12 → 13.49 | 219.48 → 226.02 |
| profiles-overlay | pandas | 687.54 / 607.07 | 1.133 | 30.39 → 30.38 | 233.52 → 232.79 |
| profiles-overlay | polars | 647.36 / 511.18 | 1.266 | 30.61 → 31.36 | 242.12 → 245.38 |
| profiles-overlay | pyarrow | 666.37 / 524.70 | 1.270 | 31.07 → 31.64 | 236.30 → 243.03 |

All 30 before/after figure pairs match at 1e-12. Every measured median plot time improved in these cases, but **overlay memory regressed**: Polars additional RSS rose about 0.75 MiB and Arrow about 0.57 MiB. Those regressions remain visible alongside the timing gains. The raw file contains all 60 trials, quartiles, baselines, peaks, figure hashes and library fingerprints.

The six current CI cold/warm sklearn/LightGBM importance reports have zero backend parity error and match this source fingerprint. Earlier million-row model-explanation timings still belong to 080f946b. Their pandas PDP/conditional computation and importance regressions, and wide-frame Polars memory overhead, remain unresolved; these plotting results do not supersede them.

## Remaining migration work at this checkpoint

Aspect/triplot and fairness plotting still need native preparation. Older private pandas algorithms need consolidation, and strict conversion guards need extension through remaining families. Incompatible mixed scalar/large-integer plot axes require separate selections rather than silent coercion. Pandas public views and scalar/index formatting, and NumPy/SciPy/tensor contracts, are intentional boundaries. Lazy/distributed/GPU engines, multiclass and arbitrary objects/concurrency remain uncertified.

Python >=3.10 is proposed for maintainer review. Official pandas/PyPI list 3.1.0 as current stable (October 22, 2026); no pandas 4 release is published. The <4 cap is precautionary future-major policy, not a demonstrated pandas 4 incompatibility. Pandas4Warning names future deprecations, not the installed version.

PR [#587](https://github.com/ModelOriented/DALEX/pull/587) remains draft and unmerged. The upstream default branch was not modified. [Raw trials](native-cp-plotting-followup/plot-benchmark.json), [CI audit](native-cp-plotting-followup/ci-validation.json), [artifact fingerprints](native-cp-plotting-followup/manifest.json).
