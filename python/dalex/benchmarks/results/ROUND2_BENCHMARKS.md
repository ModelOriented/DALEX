# Second refinement benchmarks

Latest sampled-workflow library: [03545055](https://github.com/brunocarlin/DALEX/commit/035450550ee922f07a3155cc1675ef6ac73ce9c9). Ordinary Windows Python 3.14.4, pandas 2.3.3, Polars 2.1.0. One million original Polars rows; sample 1000; B3; five trials/backend; one warm-up inside every isolated worker. Same saved fitted model/data/seeds/settings across backend pairs. Model/BLAS/OpenMP threads 1; Polars explicitly 8. These are not equal-total-core comparisons. Original Polars construction/training and process startup excluded; pandas conversion included. No other local tests/benchmarks ran during these timings. RSS sampled every 2 ms may miss peaks.

Ratios are pandas median time / Polars median time. Above 1 favors Polars. Small differences and comparisons with earlier sessions do not establish a causal code improvement.

| Model | Workflow | pandas/Polars total seconds | Total ratio | Explanation-only ratio | pandas/Polars peak RSS MiB |
|---|---|---:|---:|---:|---:|
| sklearn | importance | 0.0955/0.0829 | 1.152 | 1.070 | 460.0/384.8 |
| sklearn | breakdown | 0.0434/0.0394 | 1.100 | 0.923 | 459.7/387.6 |
| sklearn | shap | 0.0599/0.0526 | 1.139 | 1.003 | 460.4/387.7 |
| sklearn | partial | 0.0490/0.0449 | 1.091 | 0.937 | 477.5/403.0 |
| sklearn | accumulated | 0.1341/0.1274 | 1.053 | 0.997 | 482.7/406.4 |
| sklearn | conditional | 0.0559/0.0524 | 1.067 | 0.938 | 476.9/401.0 |
| lightgbm | importance | 0.0887/0.0780 | 1.137 | 1.034 | 461.3/387.1 |
| lightgbm | breakdown | 0.0395/0.0311 | 1.268 | 1.028 | 461.4/389.4 |
| lightgbm | shap | 0.0518/0.0435 | 1.189 | 1.018 | 461.3/390.4 |
| lightgbm | partial | 0.0501/0.0452 | 1.110 | 0.942 | 480.1/401.4 |
| lightgbm | accumulated | 0.1345/0.1262 | 1.066 | 1.005 | 483.8/406.7 |
| lightgbm | conditional | 0.0565/0.0506 | 1.115 | 0.961 | 480.5/404.0 |

Every measured numerical parity error is zero. Some explanation-only stages are slower; native input preservation principally avoids conversion and feature-edit costs, and does not accelerate arbitrary model inference.

## Wide-frame before/after experiment

128 rows, 1000 features, N32, B3,three trials/backend; identical saved models across source revisions. Baseline 89e8db5c precedes removal of redundant native copies; candidate 42b94c7f includes that change and batched validity checks. Later metadata/JSON fixes are outside these finite workloads. Fixed baseline-before-candidate order and small trial count limit causal interpretation. [Source manifest](round2/wide-source-manifest.json) resolves the candidate CLI's literal HEAD argument without rewriting raw data.

| Model | Workflow | Baseline pandas/Polars seconds | Candidate pandas/Polars seconds | Candidate ratio |
|---|---|---:|---:|---:|
| sklearn | breakdown | 4.4790/1.2647 | 4.4392/1.1868 | 3.740 |
| sklearn | shap | 6.5012/1.8615 | 6.3828/1.7851 | 3.576 |
| lightgbm | breakdown | 3.1435/2.0150 | 3.0048/1.9581 | 1.535 |
| lightgbm | shap | 4.4825/2.9151 | 4.4261/2.8759 | 1.539 |

Polars medians decreased about 1–6%, but pandas timings also changed. This does not prove that all differences came from clone removal. Wide-frame Polars RSS remains higher in the measured cases.

## Earlier evidence and regressions

The earlier [refinement report](REFINEMENT_BENCHMARKS.md) retains its own source snapshots and all trials. Its sampled-workflow ratios 1.26–2.02 are larger than this session's measured ratios; they are not reassigned to the latest source or advertised as universal gains. Its separate full-million-row importance experiment (three trials/backend, inference dominant) remains historical evidence, including all memory results. This session does not establish why absolute timings changed between sessions.

Raw reports in [round2/](round2/) preserve timings, quartiles, min/max, per-trial memory, fitted-model hashes, versions and numerical errors.
