# Aggregated plotting checkpoint: 4818d9f

This records the validated aggregate implementation before the subsequent CP plot migration. Its measurements and checks remain tied to the commits below.

Final implementation: [4818d9f](https://github.com/brunocarlin/DALEX/commit/4818d9fa644a9fe5769a28de3b4469297d2bafa1).
Native aggregate plotting: [1239495](https://github.com/brunocarlin/DALEX/commit/123949588a46410bea67d8e5a7b1f5beec70a18d);
nullable label fix: [1ad9793](https://github.com/brunocarlin/DALEX/commit/1ad9793a0d78c0b836a94e961d3b2f58168b6e3a).
The earlier plotting report and raw benchmarks are committed in
[6835516](https://github.com/brunocarlin/DALEX/commit/6835516).

## Implemented and validated

Aggregated partial/conditional/accumulated line and bar plots now prepare native tables.
Legacy pandas results remain writable and authoritative after access. Group captions
preserve nullable numeric promotion, categorical and missing values, nanoseconds,
timezones, durations and backend-specific date formatting. Temporal caption formatting
uses pandas scalars, not pandas tables. Raw CP overlays remain an explicit unfinished path.

- 99 new focused cases, including native fit-to-plot construction guards and complete figure comparison.
- Final source: 1,421 passed / 18 skipped / zero failures in both complete local primary Python 3.14 and pandas-nightly suites.
- [Final CI](https://github.com/brunocarlin/DALEX/actions/runs/41152948986): all 21 jobs passed. Audited artifacts: 15 full suites (each 1,421/18), eight PyTorch cases, eight TensorFlow cases, and 18 optional PPS cases in their compatible environment.
- All 18 ordinary-suite skips are optional legacy PPS tests. No DALEX test failure remains in these runs.
- 83 complete figures match unmodified upstream at rtol=atol=1e-12; 27 numerical outputs match, maximum CI absolute error 8.88e-16. Fifteen real upstream pickle objects load.
- Python 3.14t passes with actual GIL disabled after all 21 dependency imports. The unforced import probe still identifies SHAP's automatic GIL fallback; this is an upstream dependency limitation, not proof of arbitrary extension concurrency safety.
- Normalized library SHA256: `22dca5874b55e8d01121cdd118593f3d52533d2ee28a0ff651d5d8f53daf7f7d`.
- Earlier superseded CI attempts were cancelled by workflow concurrency after corrective commits. They are not counted as successful validation of this head.

## Pandas correction

The [official pandas site](https://pandas.pydata.org/) and [PyPI](https://pypi.org/project/pandas/)
list **3.1.0**, released October 22, 2026. No pandas 4 release is published.
`<4` is a precautionary future-major dependency bound, not a demonstrated incompatibility.
The PR and repository now say this explicitly. Development coverage uses pandas 3.2.
[`Pandas4Warning`](https://pandas.pydata.org/docs/dev/reference/api/pandas.errors.Pandas4Warning.html)
is a warning emitted by pandas 3.x about a future change; it does not identify the installed version.
CP compatibility code still emits the convert_floating deprecation on modern pandas.
Python >=3.10 remains a proposed compatibility change for maintainer review.

## Performance evidence and limits

No aggregate-plot speedup is claimed: these new changes have not been benchmarked.
The [preceding source-stamped plotting report](https://github.com/brunocarlin/DALEX/blob/python-narwhals-backends/python/dalex/benchmarks/results/NATIVE_PLOTTING_FOLLOWUP.md)
records Polars ROC plotting 1.78x and Shapley plotting 1.51x faster than its reference,
with pandas importance plotting about 5% slower. It includes all 120 trials, RSS samples
and matching figures. Those measurements belong to cec2f25, not this final aggregate implementation.
The earlier model-explanation report still records pandas PDP/conditional computation
about 45–50% slower and importance about 20% slower, plus higher wide-frame Polars RSS.
The six final CI cold/warm importance reports match the current source fingerprint and
have zero measured backend parity error. They do not establish general speed or memory improvements.

## Work still required for a full transition

1. CP plot preparation and the aggregate raw-profile overlay: retain typed partitions,
   observation indexes, hover data and edited pandas views. Remove its deprecated conversion flag.
2. Aspect/triplot and fairness plot preparation: preserve dendrogram geometry, subgroup
   ordering, privileged-group handling, missing scores and all interactive trace payloads.
3. Consolidate duplicate private pandas Break Down, Shapley and aggregate algorithms
   into adapters over common kernels; retain historical imports where compatibility requires them.
4. Extend strict native fit-to-plot guards to those remaining families. Improve plotting of
   incompatible mixed-type/large-integer native partitions without rounding or whole-table conversion.
5. Profile the recorded explanation-time and memory regressions, then rerun identical-model,
   identical-data/seed/settings benchmarks against pinned upstream and across pandas/Polars.

This is **not yet a full internal Narwhals transition**. Writable pandas APIs and numerical
NumPy/SciPy/tensor contracts are intentional. Pandas-free packaging, lazy/distributed/GPU
engines, multiclass and arbitrary object/concurrency cases remain separate uncertified scope.

PR [#587](https://github.com/ModelOriented/DALEX/pull/587) remains draft, open and unmerged.
The upstream master remains `77ea3ad9580a3b68aaa94d8dca75d11bd527848f`.
Working checkout: `D:/CodexScratch/DALEXNarwhals-20261030/validation-checkout`.
Raw final audit and local XML files are retained in its `.local-dist` directory; CI artifacts
are attached to the linked run. No upstream merge or default-branch write was performed.
