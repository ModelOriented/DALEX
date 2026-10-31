# Migration evidence

Start with the [final migration review](NARWHALS_FINAL_REVIEW.md) for the latest
validated source, CI, numerical precision, benchmarks and remaining limits.
Earlier checkpoints include [shared-kernel consolidation](SHARED_KERNEL_CONSOLIDATION.md)
and the source-stamped [Aspect](ASPECT_PLOTTING_FOLLOWUP.md),
[fairness](FAIRNESS_PLOTTING_FOLLOWUP.md) and [CP](CP_PLOTTING_FOLLOWUP.md) checkpoints.
They separate local validation, exact-commit CI, dependency limitations and
plotting-only measurements. The [end-to-end benchmark tables](FULL_MIGRATION_BENCHMARKS.md)
belong to source 080f946b; later plotting results do not relabel those timings.

Raw JSON is generated measurement output. GitHub collapses it by default in the
PR diff; it remains available to expand or download. Every trial and its summary
statistics are retained. The `round2/` directory retains earlier experiments;
`round2-validation.json` records source/package/CI validation. The `refinement/`
directory retains the preceding experiments and standalone dependency probes.

Earlier reports in this directory are historical, with their own source commits
and dependency versions. In particular, `NATIVE_ESTIMATOR_*`, `PHASE2_VALIDATION`,
`REVIEW_READINESS` and `VALIDATION` describe earlier implementations and test
counts. Their measurements are not silently reassigned to the latest source.

The benchmark entry points are [bench_existing_polars.py](../bench_existing_polars.py)
(data already held in Polars), [bench_variable_importance_backends.py](../bench_variable_importance_backends.py)
(importance and independent dataframe/inference attribution), and
[profile_backend_overhead.py](../profile_backend_overhead.py). Run them
from the package directory, or use repository-relative paths from its root.
Fit models and construct original inputs outside timing only where the stated
protocol explicitly excludes them. Keep benchmark workers isolated from test
jobs, fix threads/seeds/settings, and report regressions alongside gains.
