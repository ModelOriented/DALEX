# Full migration benchmarks

Library source: [080f946b](https://github.com/brunocarlin/DALEX/commit/080f946b4bb3f53134957875d8e7b3fbd854d71c). Normalized Python source SHA256: `a06cae644c9ed0b5df77fb70f80aeaa5c1ee8c34edcf31b9caba32f328c4f31d` (131 modules).

Ordinary Windows Python 3.14.4; pandas 2.3.3, Polars 2.1.0, Narwhals 2.28.0. Identical saved fitted sklearn/LightGBM models, values, targets, seeds and explanation settings within each comparison. Five measured trials per backend, isolated spawned workers, alternating backend order, and an excluded warm-up inside every worker. Model/BLAS/OpenMP threads 1; Polars threads 8. This is not an equal-total-core comparison. No other DALEX tests or benchmarks ran during these timings; ordinary desktop background activity was not controlled.

Original Polars frame creation, training, model loading and process startup are excluded. The pandas route includes conversion of the original frame. Both routes publish the compatible pandas explanation result. Time entries are median seconds [25th, 75th percentile]. Speed ratios are pandas/Polars; above 1 favors Polars. RSS is sampled every 2 ms and may miss peaks. Absolute process RSS includes runtimes and warm allocator caches; the increase above the warm baseline is reported separately.

## One million existing Polars rows; ten features; sample 1,000; B=3

| Model | Method | pandas seconds [IQR] | Polars seconds [IQR] | Total ratio | Explanation-only ratio | Peak RSS pandas/Polars MiB | RSS increase pandas/Polars MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| sklearn | importance | 0.1349 [0.1341, 0.1353] | 0.0825 [0.0815, 0.0828] | 1.635 | 1.244 | 458.8/385.9 | 3.8/3.9 |
| sklearn | breakdown | 0.0693 [0.0684, 0.0703] | 0.0365 [0.0363, 0.0372] | 1.898 | 0.984 | 458.5/385.6 | 3.8/4.0 |
| sklearn | shap | 0.0834 [0.0820, 0.0842] | 0.0508 [0.0503, 0.0513] | 1.642 | 0.998 | 458.9/386.6 | 3.8/4.0 |
| sklearn | partial | 0.0890 [0.0887, 0.0897] | 0.0328 [0.0323, 0.0332] | 2.716 | 1.732 | 468.1/401.1 | 11.3/5.1 |
| sklearn | accumulated | 0.1027 [0.1022, 0.1033] | 0.0399 [0.0398, 0.0401] | 2.572 | 1.762 | 468.2/403.4 | 11.1/5.0 |
| sklearn | conditional | 0.0977 [0.0972, 0.0979] | 0.0361 [0.0360, 0.0361] | 2.709 | 1.819 | 467.9/402.3 | 10.7/6.0 |
| lightgbm | importance | 0.1312 [0.1303, 0.1313] | 0.0823 [0.0802, 0.0842] | 1.595 | 1.202 | 460.8/388.2 | 3.8/3.9 |
| lightgbm | breakdown | 0.0635 [0.0629, 0.0639] | 0.0327 [0.0315, 0.0329] | 1.942 | 0.955 | 461.0/387.5 | 3.8/4.1 |
| lightgbm | shap | 0.0756 [0.0740, 0.0761] | 0.0418 [0.0399, 0.0429] | 1.810 | 1.019 | 461.1/388.0 | 3.8/4.1 |
| lightgbm | partial | 0.0884 [0.0877, 0.0887] | 0.0329 [0.0320, 0.0330] | 2.686 | 1.706 | 470.2/401.0 | 11.6/4.3 |
| lightgbm | accumulated | 0.1023 [0.1018, 0.1023] | 0.0396 [0.0391, 0.0396] | 2.586 | 1.760 | 470.2/404.6 | 11.5/5.7 |
| lightgbm | conditional | 0.0994 [0.0988, 0.0996] | 0.0371 [0.0367, 0.0374] | 2.676 | 1.813 | 470.0/403.0 | 11.5/5.4 |

Maximum pandas/Polars absolute numerical difference: **8.88e-16**; comparison tolerance rtol=atol=1e-10. Upstream numerical compatibility is checked separately by 27 reference outputs; timings alone do not certify numerical parity.

## Pandas regressions against unmodified upstream

The current benchmark driver reuses the exact saved estimator against upstream 77ea3ad9. The unchanged upstream baseline was measured before the final optimized source; source versions were not alternated, so small before/after differences are observational. Candidate/upstream time ratios above 1 indicate regressions. The final column compares the native Polars workflow with the upstream pandas workflow.

| Model | Method | Current pandas / upstream pandas total seconds | Candidate/upstream total ratio | Candidate/upstream explanation ratio | Upstream pandas/current Polars total ratio |
|---|---|---:|---:|---:|---:|
| sklearn | importance | 0.1349/0.1203 | 1.121 | 1.198 | 1.459 |
| sklearn | breakdown | 0.0693/0.0777 | 0.892 | 0.817 | 2.127 |
| sklearn | shap | 0.0834/0.0946 | 0.881 | 0.847 | 1.864 |
| sklearn | partial | 0.0890/0.0727 | 1.225 | 1.495 | 2.218 |
| sklearn | accumulated | 0.1027/0.1568 | 0.655 | 0.576 | 3.926 |
| sklearn | conditional | 0.0977/0.0784 | 1.245 | 1.484 | 2.176 |
| lightgbm | importance | 0.1312/0.1172 | 1.120 | 1.200 | 1.424 |
| lightgbm | breakdown | 0.0635/0.0760 | 0.836 | 0.777 | 2.322 |
| lightgbm | shap | 0.0756/0.0863 | 0.876 | 0.836 | 2.067 |
| lightgbm | partial | 0.0884/0.0732 | 1.208 | 1.454 | 2.224 |
| lightgbm | accumulated | 0.1023/0.1587 | 0.644 | 0.559 | 4.013 |
| lightgbm | conditional | 0.0994/0.0805 | 1.235 | 1.486 | 2.167 |

## Wide inputs: 128 rows, 1,000 features, sample 32, B=3

| Model | Method | pandas seconds [IQR] | Polars seconds [IQR] | Total ratio | Peak RSS pandas/Polars MiB | RSS increase pandas/Polars MiB |
|---|---|---:|---:|---:|---:|---:|
| sklearn | breakdown | 4.0661 [4.0642, 4.0866] | 1.0742 [1.0639, 1.1171] | 3.785 | 228.8/230.2 | 0.5/0.6 |
| sklearn | shap | 6.0320 [6.0308, 6.0725] | 1.6990 [1.6880, 1.7163] | 3.550 | 228.8/233.8 | 0.2/1.2 |
| lightgbm | breakdown | 2.9092 [2.9032, 2.9277] | 1.8888 [1.8773, 1.8915] | 1.540 | 228.9/233.4 | 0.3/2.1 |
| lightgbm | shap | 4.3178 [4.2826, 4.3469] | 2.8280 [2.7977, 2.8532] | 1.527 | 231.4/236.6 | 0.3/2.2 |

Maximum wide-input absolute numerical difference: **0**. Retain memory regressions even where native processing is faster. These workloads do not establish universal estimator or free-threading speedups.

## Profile-grid regression found during refinement

Source 650a11fe filtered reference values in a Python loop before constructing automatic numeric grids. Native null/NaN filtering and min/max reductions replace that scan; quantile grids enter NumPy as vectors. A regression test forbids exporting all reference cells to Python just to build a grid. The earlier six-family sklearn measurement is retained with its own source identity, rather than relabeled as final evidence. Source order, different checkout locations and desktop conditions limit causal timing claims.

| Method | Earlier pandas/Polars total seconds | Final pandas/Polars total seconds |
|---|---:|---:|
| partial | 1.2497/1.1909 | 0.0890/0.0328 |
| accumulated | 1.3002/1.2541 | 0.1027/0.0399 |
| conditional | 1.2524/1.2066 | 0.0977/0.0361 |

A second profile identified redundant filtering/copying of clean reference columns and per-expanded-row split checks. The final implementation skips unnecessary filters and normalizes each split once before repetition. The instrumented pandas workflow fell from 196,774 to 76,934 function calls; profiler timings are not used as benchmark speedup claims. Raw before/after profiles and source-783f0697 trial reports are retained. Five-trial measured explanation medians follow:

| Model | Method | 783f0697 pandas/Polars explanation seconds | Final pandas/Polars explanation seconds |
|---|---|---:|---:|
| sklearn | partial | 0.0742/0.0352 | 0.0566/0.0327 |
| sklearn | accumulated | 0.0866/0.0423 | 0.0702/0.0398 |
| sklearn | conditional | 0.0869/0.0396 | 0.0654/0.0359 |
| lightgbm | partial | 0.0817/0.0356 | 0.0559/0.0328 |
| lightgbm | accumulated | 0.0940/0.0423 | 0.0695/0.0395 |
| lightgbm | conditional | 0.0836/0.0382 | 0.0672/0.0370 |

## Reproduction

Use `bench_existing_polars.py` with the settings embedded in each JSON. The driver uses data seed 123, model/explanation seed 42, 20 estimator iterations, three explanation rounds and explicit thread limits. Reuse `--model-file` across source revisions. All trials, medians, quartiles, absolute/incremental RSS, model fingerprints and normalized source fingerprints are retained in `full-migration/`. The required CI additionally archives cold/warm importance benchmarks for explicit NumPy, sklearn-native and LightGBM-native callbacks.
