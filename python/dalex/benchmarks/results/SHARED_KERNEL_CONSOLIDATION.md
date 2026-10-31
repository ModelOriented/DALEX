# Shared-kernel consolidation

Older breakdown/interactions, DALEX Shapley, CP/aggregate and Aspect/triplot helpers now adapt shared native kernels. Their default pandas return types, indexes, column order, nested metadata and dtypes remain compatible. Categorical statistics share vector kernels. PPS still receives pandas at its external boundary, but reshaping/symmetrizing scores returns to the original backend. Duplicate algorithms are removed rather than retained as a second computation path.

## Correctness fixes exposed by stronger comparisons

- **Nonlinear hierarchical Shapley:** previous source d9f2e55 restarted the serial path RNG at each cut. A six-feature nonlinear upstream example differed by up to **0.4574545767390757**. One local stream now advances across cuts and final singles. Explicit seeds remain isolated from caller NumPy state. Fixed upstream expectations pass on pandas/Polars/Arrow; the independent numerical harness adds nonlinear default/Shapley hierarchies. Linear examples alone were insufficient evidence.
- **Pandas index-sensitive Aspect callbacks:** local-surrogate intervention batches again use the historical reset index, while reference samples retain sampled labels. Serial Shapley paths retain the historical index handling. Upstream regression expectations exercise callback predictions that depend on those labels.
- Consolidation initially changed floating versus object quicksort tie order in a private helper and the standalone single-variable model helper's height dtype. The runtime-matched upstream comparison caught both; their legacy contracts are restored. Fixed cross-platform NumPy tie order is not assumed.
- Residual diagnostics now require optional statsmodels only for smoothing. A minimal wheel installation exposed the unconditional check even with smoothing disabled.

## Local validation

Validated library: **135 modules**, SHA256 `f235a1468429e225ff5ac6f4895f9ab6a09003ed35882c871d2bfbe1c73ba22b` (paths/source sorted, CRLF normalized).

- Final exact-source complete suite: **1,778 passed / 18 skipped**, zero failures. The 18 skips are optional PPS cases.
- Minimum Polars 1/Narwhals 2 and pandas-nightly focused checks: **86 passed each**. These checks preceded only the conditional statsmodels import change; the final source has the complete primary suite and built-wheel validation.
- A real compatible Python 3.10/pandas 1.5.3/PPS 1.3 environment passes **32 selected cases**, including all 18 original PPS tests. Actual PPS matrices match across pandas/Polars/Arrow with zero difference and retain native storage. PPS's deprecated pkg_resources usage and pandas<2 dependency remain upstream constraints.
- **48 added real sklearn/LightGBM regression/classification integrations** cover Polars/Arrow Aspect fitting, exports, figures and pickle round trips with temporal/categorical inputs. Fourteen additional regression/boundary cases cover index handling, nonlinear paths and PPS matrix restoration.
- **106 private-helper outputs** match unmodified upstream, including indexes, dtypes and heterogeneous cells. The old reference receives the documented NumPy spelling shim, a pickle-safe residual callback and matching global seed initialization where old private helpers relied on constructor-managed global state.
- **29 numerical outputs** match at rtol=atol=1e-12; local maximum absolute error **4.44e-16**. **174 figure comparisons** pass: 168 exact payloads and six triplots with redundant unlabelled collinear hover samples normalized. **15 actual upstream pickle objects** load.
- The built wheel passes **12 pandas-only core workflows**, including fitting, writable results, exports, plots and pickle, without Polars, Arrow, sklearn, SHAP, LIME or statsmodels installed. Python 3.14.4, pandas 3.1.0, Plotly 7.1.0, SciPy 1.18.1. Wheel source matches the exact fingerprint above. A required CI job repeats this packaging check.

Speculative future-major pandas/Narwhals runtime caps have been removed. Minimums remain pandas 1.5.3 and Narwhals 2.0; code uses stable.v2. This is a packaging policy, not certification of unreleased versions. See [Narwhals' stable-API policy](https://narwhals-dev.github.io/narwhals/backcompat/) and [Python packaging guidance on known bounds](https://packaging.python.org/en/latest/discussions/install-requires-vs-requirements/). Python >=3.10 remains proposed for maintainer review, with 3.14 primary.

## CI and remaining work

The preceding [Aspect checkpoint](ASPECT_PLOTTING_FOLLOWUP.md) passed all 21 CI jobs, but that did not expose the nonlinear RNG error. Consolidation [fdb23b0](https://github.com/brunocarlin/DALEX/commit/fdb23b037e088bb6b9b6e72b2977cce6a04831ae) passed **[all 22 exact-commit CI jobs](https://github.com/brunocarlin/DALEX/actions/runs/41166813766)**. Downloaded artifacts verify 15 full suites (1,778/18 each), eight PyTorch cases, eight TensorFlow cases, 18 PPS cases, 29 numerical outputs (maximum error 8.88e-16), 106 helper outputs, 174 figures and 15 upstream pickle objects. The installed-wheel job verifies 12 pandas-only families with the same source fingerprint. Actual 3.14t GIL state is disabled after all 21 imports; unforced SHAP enables it. [CI artifact audit](shared-kernel-consolidation/ci-validation.json).

Fairness text-check preparation still performs pandas threshold reductions. Existing pandas importance/PDP/conditional regressions and wide-frame Polars RSS remain to profile and remeasure. No end-to-end performance improvement is claimed for this consolidation. `bench_source_comparison.py` supports alternating immutable sources, identical saved models/data hashes, per-trial numerical checks and sampled RSS; the forthcoming measurements must identify their own implementation.

Public writable pandas views, dtype/index/scalar formatting and explicit estimator/SciPy/tensor boundaries remain deliberate. Pandas-free packaging, untested engines, lazy/distributed/GPU execution, arbitrary objects and extension concurrency are uncertified. Unforced SHAP still enables the GIL; forced-disabled 3.14t tests do not certify arbitrary compiled-extension safety. PR #587 remains draft and unmerged.

[Raw local validation](shared-kernel-consolidation/local-validation.json) · [Helper contracts](shared-kernel-consolidation/upstream-adapters.json) · [Numerical parity](shared-kernel-consolidation/upstream-numerical.json) · [Installed wheel](shared-kernel-consolidation/pandas-only-wheel.json) · [Evidence fingerprints](shared-kernel-consolidation/manifest.json).
