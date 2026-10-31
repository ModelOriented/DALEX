# Final migration audit and free-threaded validation

This is the earlier audit snapshot. Expanded native-estimator, Arrow, neural,
seeded-concurrency and performance evidence is in
[NATIVE_ESTIMATOR_VALIDATION.md](NATIVE_ESTIMATOR_VALIDATION.md).

Implementation source: `107348aa` (the following documentation/evidence commit
does not change executable code).

[Final CI run](https://github.com/brunocarlin/DALEX/actions/runs/40593699475):
**all 14 jobs passed**. Each of the 12 full-suite jobs reports **142 passed,
22 skipped, zero failures/errors**. Minimum-pandas/PPS and cold/warm benchmarks
also passed. The 3.14t job is required, with no continue-on-error exception.
`review-validation.json` records per-job links/JUnit counts and distribution hashes.

## Audit findings and fixes

- Mixed NumPy callback matrices now reconstruct numeric columns separately and
  restore categorical metadata. Lists used as native observations use the same
  bridge. Numeric continuous perturbations are not cast back to integers.
- Arena now compares complete pandas/Polars numerical payloads for all 13 plot
  families. Seeds/settings are identical. The single normalized presentation field
  is the ceteris-paribus observation ID (pandas string index versus native row
  position); numerical values retain a 1e-10 tolerance.
- Nullable categorical/string/integer metadata and all-null numeric columns no
  longer attempt to sort missing sentinels. Eight backend cases ensure extraction
  does not fill or mutate the original model features.
- Pandas 3 fixes: fairness masks mutate an owned array, fairness reshaping retains
  row-major order and missing scores without the removed stack argument, Arena
  accepts StringDtype groups, and breakdown plot labels remain strings. An old
  fairness test now explicitly uses positional `.iloc[0]`.
- Sampling/replacement helpers preserve input immutability, pandas row labels,
  categorical schemas and positional/repeated-row semantics. Native-only callback
  tests cover the migrated families; external NumPy and pandas statistics/result
  boundaries remain explicit.

## Resolving the 3.14t installation blocker

The original PyPI-only route could not install `polars-runtime-32`, and pandas<3
also had no matching cp314t wheel. The committed Pixi manifest/lock installs the
free-threaded interpreter and Polars from conda-forge; pip installs the remaining
test dependencies and DALEX, with pandas>=3,<4. No source compilation or wheel
tag alteration is used. Regular pandas users do not need Pixi.

Local Windows validation uses CPython 3.14.10 free-threaded, Polars 2.1.0,
pandas 3.1.0, NumPy 2.5.4, SciPy 1.18.1, sklearn 1.9.1 and SHAP 0.54.1.
`pip check` passes. `verify_runtime.py` checks Py_GIL_DISABLED and the actual GIL
state after every scientific dependency import with PYTHON_GIL=0/EXPECT_GIL=0.

The separate no-override import probe found **SHAP's shap._cext automatically
enables the GIL**. Its extension has not declared free-threading safety; the other
14 tested imports retained the disabled GIL on both Windows and Linux. The JSON
probe preserves that warning. Full tests with PYTHON_GIL=0 measure explicitly
disabled-GIL execution, not proven safety of concurrent third-party extension
calls. No upstream extension safety flags are changed and no free-threading
speedup is claimed.

The complete local Windows 3.14t suite passed: **142 passed, 22 skipped** in
190.10 seconds (191 warnings, including pandas 4 deprecations outside the proposed
pandas<4 bound). The required Linux 3.14t job also passed **142/22**, zero failures
or errors, in 307.72 seconds. Its runtime log reports Py_GIL_DISABLED=True and
GIL enabled=False after all imports. These timings come from different machines
and are not a performance comparison.

## Distribution and numerical evidence

Both wheel and source archive build. A fresh regular Python 3.14 environment
installs the wheel and full dependencies. Backend, integration and remaining-family
tests are copied to a separate directory; the import is asserted to resolve to
the installed `site-packages/dalex`, not the source checkout. Dependency checking
passes. Exact final test totals are recorded in `review-validation.json`.
The audited installed-wheel run passed **82 tests**, zero failures/skips, in
46.06 seconds. Earlier wheel runs were repeated after executable audit fixes.

The fresh unmodified-upstream comparison still has maximum absolute error **0**
across all 17 numerical output groups (prediction/residuals, metrics/diagnostics,
grouped weighted permutation variants, breakdown/interactions/Shapley values,
ceteris-paribus and aggregated profiles). SHAP/LIME, surrogate and Aspect tests
compare backend outputs directly; Arena now compares payloads instead of only
checking that a plot was produced.

Backend performance claims remain scoped to the source-stamped ordinary Python
3.14 measurements. [PHASE2_VALIDATION.md](PHASE2_VALIDATION.md) retains earlier
Windows/Linux observations, including regressions. The final CI benchmark uses
Python 3.14.8, three trials/two rounds, identical fitted estimator, data, weights,
seeds and settings, ten features and a 10k sample cap:

| Rows | Cold pandas/Polars seconds | Warm pandas/Polars seconds | Warm ratio | Warm peak RSS pandas/Polars MiB |
|---:|---:|---:|---:|---:|
| 1k | 0.0657 / 0.0705 | 0.0600 / 0.0560 | 1.071 | 250.5 / 276.9 |
| 10k | 0.3276 / 0.3253 | 0.3231 / 0.3113 | 1.038 | 255.2 / 279.9 |

Cold ratios are 0.932/1.007; ratios above 1 favor Polars. The 1k cold regression,
near-parity 10k cold result, small warm gains and higher Polars RSS are all
reported. Maximum numerical parity error is 0 in both modes. Individual trials,
IQRs, package versions and RSS samples are archived in
`linux-python314-review-{cold,warm}.{json,md}`. Training/process startup are
excluded; native construction/conversion/inference are timed. RSS sampling at
2ms may miss short peaks. These host-specific measurements do not establish
any universal improvement or free-threading speedup.

## Upstream and review scope

After fetching upstream, `upstream/master` is still
`77ea3ad9580a3b68aaa94d8dca75d11bd527848f`: no new upstream commits need integration.
The contribution proposes Python>=3.10, Narwhals>=2,<3 and pandas>=1.5.3,<4;
maintainers should review those bounds. Pandas remains a required statistics and
result-table dependency. Polars is optional for pandas-only users.

The supported CI matrix retains all ten regular Python 3.10–3.14 Linux/macOS
pandas 2 entries, adds a regular 3.14/pandas 3 entry, and requires the 3.14t Linux
job. Minimum pandas 1.5.3/Narwhals 2.0 and optional PPS have a separate job. The
22 inherited skips remain visible (18 Aspect/PPS; four old Ubuntu-only Arena
tests); newer integration tests independently exercise these methods.

Remaining limits: lazy/other native backends, TensorFlow/PyTorch Deep/Gradient
SHAP and arbitrary model libraries are uncertified. PPS still requires the
separate legacy pandas<2 environment. DALEX retains global NumPy RNG use, so
simultaneous seeded explanation calls are not certified reproducible. Free-threaded
SHAP concurrent-call safety remains an upstream concern even when this suite passes.
Detailed performance/memory comparisons cover permutation importance; other
explanation families still need benchmarks at scale. Dtype tests cover selected
nullable/categorical/date adapter cases, not every method with timezone-aware,
decimal or nested columns. Explanation results/statistics still require pandas;
this contribution does not remove that dependency. Additional Narwhals backends
need their own integration matrix before being advertised as supported.
PR #587 stays draft; upstream Actions still require maintainer approval. Nothing
is merged and the upstream base branch is unchanged.
