# Initial migration validation evidence

This is the historical first-stage report. See [PHASE2_VALIDATION.md](PHASE2_VALIDATION.md) for the remaining-family migration and cold/warm attribution.

Measured implementation: `0cbe570e5013cdd0b931c17a36243eab5b266021`.
The result files name their measured source revisions; documentation-only commits
may follow without changing the measured implementation.

- Windows Python 3.14.4 full Python suite: **116 passed, 22 skipped**, 77 warnings.
- Python 3.14.4 backend suite: **56 passed**. Python 3.12 with Narwhals 2.0.0: **56 passed**.
- Python 3.10.20, pandas 1.5.3, NumPy 1.24.4, Narwhals 2.0.0 adapter suite: **17 passed**.
- Upstream pandas comparison: 17 numerical outputs, maximum absolute difference **0**.
- Wheel/sdist builds, installed-wheel native gather/import, compile and diff checks pass.

The inherited skips comprise 18 Aspect tests disabled for ppscore dependencies and
four Arena server tests guarded by `sys.platform.startswith('ubuntu')`, which also
skips on ordinary Linux. No new backend tests skip for missing Polars.

[CI for the measured implementation](https://github.com/brunocarlin/DALEX/actions/runs/40573180652)
archives test/dependency evidence for Python 3.10–3.14 on Linux and macOS, plus
minimum-pandas checks. Refer to that run for individual job status and interpreter
versions. An earlier implementation also passed all ten regular full-suite jobs
(109 passed / 22 skipped each); superseded workflows were canceled during iteration.

## Python 3.14t dependency blocker

The Linux free-threaded binary passes `Py_GIL_DISABLED` and disabled-GIL assertions
before installation. Dependency resolution fails because **polars-runtime-32 has
no matching wheel for this interpreter**. See the archived installation log and
`python314t-dependency-blocker.txt`. DALEX tests and the post-import GIL check did
not run there. Only the free-threaded experiment permits job failure; this is not
counted as passing DALEX coverage. On local Windows, pandas 2.3.3 source installation
also failed due to missing MSVC build tools, before DALEX tests.

## Benchmarks

The committed JSON contains every trial, package versions, inference timings,
absolute/incremental RSS, medians and IQR. The Markdown tables summarize them.

- Windows 3.14.4: 1k/10k/100k rows, 10 features, five trials, three rounds, 10k sample cap.
- Linux CI 3.14.8: 1k/10k rows, ten features, three trials, two rounds.
- One fitted estimator, identical data/weights/seeds/settings within each comparison;
  all result and raw permutation differences were **zero**.
- Polars is **slower end to end** in all these measurements. The dataframe-only
  1k experiment improves; larger dataframe-only measurements regress.
- Polars has higher sampled process RSS on both machines. RSS sampling every 2ms
  may underestimate peaks and includes the interpreter/dependencies.
- Excluded warm-up trials use fresh processes: they warm OS caches, not persistent
  worker state. Training and process startup are excluded; native construction and
  model-boundary conversion are included. These results are not general guarantees.

This increment does not certify lazy/other eager backends, external SHAP wrappers,
LIME, model surrogates, Aspect, Arena, or regression fairness. Pandas result tables
remain intentional boundaries. Python >=3.10 is a proposal requiring maintainer
review before release.
