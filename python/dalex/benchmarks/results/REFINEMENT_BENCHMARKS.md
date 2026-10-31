# Refinement benchmarks

Implementation source for the final wide/million-row runs: [b3725568](https://github.com/brunocarlin/DALEX/commit/b3725568bec9ecdef8a4329420d1c3ed0a52f462). Later missing-profile/fairness/widget changes are outside these workloads.

Windows 11, ordinary Python 3.14.4; pandas 2.3.3, Polars 2.1.0, Narwhals 2.28.0, NumPy 2.5.4, sklearn 1.9.1 and LightGBM 4.7.0. Three trials per backend, three permutation/path rounds; estimator/BLAS/OpenMP threads limited to one. Polars and Arrow conversion retain their engine defaults; the early snapshots did not record their pool sizes. These are not equal-total-core comparisons. Sequential isolated workers alternate backend order and warm inside each measured process. Training and process startup are excluded. No other local test or benchmark jobs ran during these timings.

Ratios are pandas median time / Polars median time; values above one favor Polars. Raw JSON preserves every trial, quartiles, minimum and maximum.

## Existing wide Polars input

128 rows, 1,000 features; explanation sample 32; break-down and DALEX Shapley values. Original Polars construction/training are excluded; pandas conversion, Explainer initialization and explanation are included. The same saved fitted model is reused across backends **and across the wide-frame source revisions**. Hashes are in every raw report.

| Model | Family | pandas seconds | Polars seconds | Ratio | pandas / Polars peak RSS MiB | Max numeric error |
|---|---|---:|---:|---:|---:|---:|
| sklearn | breakdown | 4.2244 | 1.1785 | 3.585 | 225.8 / 232.0 | 0 |
| sklearn | shap | 6.2529 | 1.6940 | 3.691 | 224.5 / 231.3 | 0 |
| lightgbm | breakdown | 3.0337 | 2.0518 | 1.479 | 227.1 / 234.0 | 0 |
| lightgbm | shap | 4.4602 | 3.0232 | 1.475 | 226.5 / 233.6 | 0 |

### Refinement history and pandas regression check

The e10 baseline collected the full portable schema for each one-column intervention, creating quadratic metadata work as feature count grows. 3f inspects only the replaced columns. An intermediate unconditional Polars clone workaround protected old releases but added overhead on modern wide frames; b372 limits it to Polars 0/1. These distinct stages are archived, rather than treating the combined change as a single causal measurement.

| Model / backend | Family | e10 seconds | 3f seconds | 1ac seconds | b372 seconds |
|---|---|---:|---:|---:|---:|
| sklearn / pandas | breakdown | 4.3116 | 4.3020 | 4.2124 | 4.2244 |
| sklearn / pandas | shap | 6.3760 | 6.4066 | 6.3926 | 6.2529 |
| sklearn / polars | breakdown | 2.6251 | 1.1495 | 1.4120 | 1.1785 |
| sklearn / polars | shap | 3.8829 | 1.7248 | 2.3301 | 1.6940 |
| lightgbm / pandas | breakdown | 3.0791 | 3.0628 | 3.0054 | 3.0337 |
| lightgbm / pandas | shap | 4.5872 | 4.5025 | 4.4460 | 4.4602 |
| lightgbm / polars | breakdown | 3.4237 | 2.0821 | 2.3264 | 2.0518 |
| lightgbm / polars | shap | 5.1110 | 3.0009 | 3.6718 | 3.0232 |

The unmodified upstream pandas implementation was also run with the same saved sklearn model, data and settings:

| Family | Upstream pandas seconds | Migration pandas seconds | Migration / upstream |
|---|---:|---:|---:|
| breakdown | 4.2017 | 4.2244 | 1.005 |
| shap | 6.1875 | 6.2529 | 1.011 |

Ratios slightly above one here are measured pandas regressions, not established statistical differences from three trials. Polars uses more peak RSS in this small-row/wide-frame workload.

## One million rows, full-size permutation importance

Ten float64 features, all one million rows sampled. The same fitted Polars-trained estimator is used within each pandas/Polars pair. Dataframe construction/conversion and explanation are timed; the dataframe-only scope measures equivalent sampling/permutation separately. Unlike the wide runs, fitted models were not saved across million-row source revisions: those use the same training data/settings/seeds, not a verified identical serialized model.

| Model | pandas / Polars seconds | Ratio | pandas / Polars inference seconds | Dataframe-only ratio | pandas / Polars peak RSS MiB | Max numeric error |
|---|---:|---:|---:|---:|---:|---:|
| sklearn | 28.0836 / 27.1550 | 1.034 | 26.5338 / 26.3102 | 5.930 | 705.8 / 642.9 | 0 |
| lightgbm | 24.7123 / 23.1433 | 1.068 | 23.1661 / 22.3496 | 6.109 | 696.2 / 544.9 | 0 |

Inference dominates these tall-data workloads, so a large dataframe-only gain produces a modest end-to-end gain. Preserving a native frame cannot accelerate the estimator itself. No universal acceleration, confidence interval from three samples, or free-threading speedup is claimed.

RSS is sampled every 2 ms and can miss peaks. It includes the interpreter, dependencies, models and input buffers; allocator warming affects both absolute and incremental measurements. Memory gains on tall data do not erase the wide-data regressions. Earlier cold/warm and six-family reports remain available in the parent directory with their own source stamps.

## One million original Polars rows, usual small explanation sample

Library source 42fd07b2; benchmark instrumentation source 4d471716. One million original rows, ten features, explanation sample 1,000; five measured trials/backend and three rounds. Six explanation families. Each worker warms internally, and the same saved fitted model is reused across backends and both Polars pool configurations. Model/BLAS/OpenMP threads remain one. Polars pools of eight (this host default) and one are explicitly recorded; Arrow conversion retains its default settings.

### Host-default Polars pool (eight threads)

| Model | Family | pandas / Polars workflow seconds | Workflow ratio | Explanation-only ratio | pandas / Polars peak RSS MiB | Max numeric error |
|---|---|---:|---:|---:|---:|---:|
| sklearn | importance | 0.1187 / 0.0800 | 1.483 | 1.083 | 458.5 / 383.7 | 0 |
| sklearn | breakdown | 0.0684 / 0.0375 | 1.822 | 0.953 | 458.4 / 385.1 | 0 |
| sklearn | shap | 0.0845 / 0.0537 | 1.575 | 0.975 | 458.7 / 385.6 | 0 |
| sklearn | partial | 0.0800 / 0.0465 | 1.719 | 1.009 | 477.2 / 410.9 | 0 |
| sklearn | accumulated | 0.1636 / 0.1296 | 1.262 | 1.015 | 486.6 / 408.7 | 0 |
| sklearn | conditional | 0.0822 / 0.0512 | 1.605 | 0.958 | 484.1 / 398.8 | 0 |
| lightgbm | importance | 0.1166 / 0.0785 | 1.486 | 1.059 | 460.0 / 386.2 | 0 |
| lightgbm | breakdown | 0.0653 / 0.0324 | 2.016 | 1.005 | 459.8 / 387.1 | 0 |
| lightgbm | shap | 0.0775 / 0.0444 | 1.745 | 1.007 | 460.0 / 387.8 | 0 |
| lightgbm | partial | 0.0765 / 0.0454 | 1.686 | 0.964 | 476.7 / 400.1 | 0 |
| lightgbm | accumulated | 0.1621 / 0.1287 | 1.260 | 0.999 | 487.5 / 417.9 | 0 |
| lightgbm | conditional | 0.0815 / 0.0507 | 1.608 | 0.956 | 477.2 / 407.3 | 0 |

Avoiding a full pandas conversion improves total workflow time in these sampled runs. Several explanation-only ratios remain below one: those Polars explanation phases are slower. Gains from conversion avoidance must not be described as uniformly faster explanation algorithms.

### Thread-pool sensitivity

| Model | Family | Polars workflow seconds: 8 / 1 threads | Polars explanation seconds: 8 / 1 threads |
|---|---|---:|---:|
| sklearn | importance | 0.0800 / 0.0768 | 0.0799 / 0.0767 |
| sklearn | breakdown | 0.0375 / 0.0357 | 0.0374 / 0.0356 |
| sklearn | shap | 0.0537 / 0.0491 | 0.0535 / 0.0490 |
| sklearn | partial | 0.0465 / 0.0449 | 0.0464 / 0.0449 |
| sklearn | accumulated | 0.1296 / 0.1267 | 0.1295 / 0.1266 |
| sklearn | conditional | 0.0512 / 0.0506 | 0.0511 / 0.0505 |
| lightgbm | importance | 0.0785 / 0.0759 | 0.0784 / 0.0758 |
| lightgbm | breakdown | 0.0324 / 0.0327 | 0.0323 / 0.0326 |
| lightgbm | shap | 0.0444 / 0.0437 | 0.0443 / 0.0436 |
| lightgbm | partial | 0.0454 / 0.0452 | 0.0453 / 0.0451 |
| lightgbm | accumulated | 0.1287 / 0.1256 | 0.1286 / 0.1255 |
| lightgbm | conditional | 0.0507 / 0.0503 | 0.0506 / 0.0502 |

Backend order alternates within trials, but the eight-thread configurations ran before the one-thread configurations. This fixed-order sensitivity experiment does not isolate pool size from every host/cache/order effect. Do not override the global Polars default based solely on these samples. [Polars documents](https://docs.pola.rs/api/python/stable/reference/api/polars.thread_pool_size.html) that POLARS_MAX_THREADS must be set before process start and generally recommends retaining its automatic default.

Reproduce from the repository root (set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS to 1 before launching Python):

```text
python python/dalex/benchmarks/bench_existing_polars.py --model sklearn --rows 1000000 --features 10 --sample 1000 --rounds 3 --repetitions 5 --families importance breakdown shap partial accumulated conditional --model-file model.pkl --source-revision <commit> --output report.json
```

Repeat with `--model lightgbm` and its own model cache. For the sensitivity experiment, launch separate processes with POLARS_MAX_THREADS set to 8 or 1, retaining the fitted cache for each model. Do not run other test/benchmark jobs during timing. Raw reports include every trial and IQR.
