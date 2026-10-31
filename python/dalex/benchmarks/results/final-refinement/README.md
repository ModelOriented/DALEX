# Final-source evidence, October 31, 2026

Read [the final review](../NARWHALS_FINAL_REVIEW.md) for interpretation and limits.
Measured library revision: `3b66481e829f581f4c2d427719b1601667973f3d`.
CI validation revision: `34cb63005cc2f0edec980d19ba863cc97ede9b17` (same library).
Normalized library SHA256:
`441d90693067367af8a8727caf0b4a639515a6aef3c39312cfc876e07780ce5e`.
Report/documentation successors can share this exact library source.

`evidence-manifest.json` records the exact bytes and SHA256 of every other file
in this folder. Verify the retained evidence offline with the standard library:

```sh
python python/dalex/benchmarks/results/final-refinement/verify_retained_evidence.py
```

This checks checksums, the ten archived reports and 570 worker records, common
source/model/data identities and the four local JUnit summaries. It does not
rerun workloads or replace the independent online CI snapshot audit below.
The folder's Git attributes preserve bytes across Windows/Linux checkouts.

## Validation

- `ci-validation.json`: audit of all 23 successful jobs and their artifacts in
  [run 41183547272](https://github.com/brunocarlin/DALEX/actions/runs/41183547272).
  `ci-artifact-manifest.json` retains public artifact IDs and archive SHA256s;
  all downloaded ZIPs matched before extraction. No temporary download URLs or
  credentials are committed.
- `local-validation.json` and four JUnit XML files: complete local suite,
  focused pandas-development checks and minimum-dependency precision checks.
  The manifest records the installed wheel's hash and Python requirement; the
  binary wheel is not included.
- `upstream-parity.json`, `final-upstream-adapters.json`,
  `final-upstream-labels-pandas31.json`, `upstream-pickle-compatibility.json`,
  `upstream-plot-parity.json`: separate-interpreter unchanged-source reference
  comparisons, with runtime spelling shims and figure exceptions documented.
- `float32-rounding-context.json`: actual values, magnitude and relative error
  for the largest instrumented native Float32 discrepancy. It supplements the
  54 precision properties in each complete-suite JUnit artifact.
- `final-pandas-only-package.json`: installed-wheel execution with the six
  native/model optional packages absent. `pps-backend-parity.json`: real PPS
  in its compatible separate legacy environment.
- `runtime-state.log`, `gil-import-probe.json`: actual forced-disabled GIL state
  after all 21 imports and isolated unforced import behavior. Passing the forced
  workload is not a blanket extension thread-safety certification.
- `pandas-boundary-inventory.json`: annotated pd-rooted call expressions,
  including chained calls. Method calls, aliases and dynamic behavior need the
  additional manual review/runtime guards; the static inventory is not a proof.

To repeat the snapshot artifact audit from the repository root:

```sh
gh run download 41183547272 --repo brunocarlin/DALEX --dir downloaded-artifacts
python python/dalex/benchmarks/results/final-refinement/audit-ci-snapshot.py \
  --run 41183547272 --head 34cb63005cc2f0edec980d19ba863cc97ede9b17 \
  --artifacts downloaded-artifacts --output repeated-audit.json
```

The audit is intentionally specific to this checkpoint's counts and schema. It
checks extracted artifacts, not the archive digests; use the public artifact
manifest to verify ZIP bytes separately. GitHub artifact retention can expire;
the selected structured results and local JUnit files are retained here.

## Performance

`measurements.zip` retains ten original `final-*.json` files with five-trial Windows comparisons of unmodified
upstream/current pandas and current Polars, with the earlier shared-kernel source
also measured in the two main reports. `benchmark-manifest.json` identifies each
report/source/model/data and each original JSON's SHA256; the ZIP is checked
against those hashes. `benchmark-summary.csv` retains every median,
quartile, min/max and phase/RSS statistic. Source/model/data fingerprints are
checked in every worker. The six `ci-*-cold/warm.json` files are smaller Linux
importance/adapter attribution trials with a distinct protocol; see the review.

Reproduce the main source comparison using immutable checkouts and the same
interpreter/environment. Example (package roots are `python/dalex` within each
checkout, not the inner library directory):

```sh
python python/dalex/benchmarks/bench_source_comparison.py \
  --source upstream=../upstream/python/dalex \
  --revision upstream=77ea3ad9580a3b68aaa94d8dca75d11bd527848f \
  --source before=../before/python/dalex \
  --revision before=fdb23b037e088bb6b9b6e72b2977cce6a04831ae \
  --source candidate=../candidate/python/dalex \
  --revision candidate=3b66481e829f581f4c2d427719b1601667973f3d \
  --pandas-only-source upstream --model sklearn \
  --model-file bench-output/sklearn.pkl --output bench-output/sklearn.json \
  --rows 1000000 --features 10 --sample 1000 --rounds 3 --trials 5 \
  --no-precalculate
```

Create `bench-output` first; repeat with `--model lightgbm` and its own model
file. Set `POLARS_MAX_THREADS=8` before imports (one for the single-thread run),
and model/BLAS thread environment variables to one. The driver also applies
`threadpool_limits(1)`. All versions and actual pool sizes are retained in the
raw reports. Identical saved model bytes are shared across compared workers;
hashes need not be identical after retraining on another software/hardware stack.

To regenerate the four-panel timing figure without executing any model, run
`python python/dalex/benchmarks/results/final-refinement/generate_figure.py`
with numpy/matplotlib installed. It verifies the archived JSON hashes first.

Wide settings are `--rows 128 --features 1000 --sample 32 --families breakdown shap`
with separate models. Default settings add `--precalculate --families importance
partial conditional`; single-thread and pandas 3.1 confirmations select
`--families importance conditional`. These follow-ups omit the `before` source.
Original Polars construction, model fitting, imports and warm-up are excluded;
conversion, new Explainer, fitting and compatible result publication are timed.

## Standalone dependency investigation

`probe_lightgbm_legacy_pipeline.py` does not import DALEX. The retained diagnostic
and NumPy 1.24.4 log reproduce the older local Windows/Python 3.10/LightGBM 4.7
training access violation. `probe_lightgbm_legacy_numpy.py` and three passing
simple-fit logs show the failure is not universal to that package combination.
The cause is unresolved; the supported older Windows integration separately
passes with LightGBM 4.6 and an explicit model-owned bridge. No upstream bug
report has been posted automatically.

The `lightgbm-final-*` files repeat the pipeline and three controls after the
isolated NumPy 1.23.5 environment passes dependency consistency checks. The
pipeline explicitly checks that DALEX is not imported before fitting; the
access violation persists and all simple controls pass. These are dependency
diagnostics, not failed tests in the passing supported CI matrix.
