# Native estimator benchmark tables

> Historical measurements. See [refinement benchmarks](REFINEMENT_BENCHMARKS.md) for the later experiments.

Source-stamped raw JSON files in this directory retain every trial and IQR.

## Native prediction, full-size permutation importance

Five measured trials/backend, three rounds, ten float64 features, 100k sample cap, warm in each worker. Models train on Polars; timed frames are constructed from identical NumPy values. Predictions receive native frames. Training/process startup are excluded. Ratios above 1 favor Polars.

| Model | Rows | pandas / Polars seconds | Ratio | pandas / Polars peak RSS MiB | Max parity error |
|---|---:|---:|---:|---:|---:|
| sklearn | 1000 | 0.0513 / 0.0433 | 1.184 | 214.6 / 217.8 | 0 |
| sklearn | 10000 | 0.3059 / 0.2998 | 1.021 | 220.1 / 225.5 | 0 |
| sklearn | 100000 | 2.8325 / 2.7764 | 1.020 | 267.0 / 269.4 | 0 |
| lightgbm | 1000 | 0.0477 / 0.0420 | 1.136 | 215.9 / 220.0 | 0 |
| lightgbm | 10000 | 0.2671 / 0.2611 | 1.023 | 220.4 / 226.3 | 0 |
| lightgbm | 100000 | 2.5441 / 2.4170 | 1.053 | 267.7 / 264.7 | 0 |

## Data already in Polars, wider explanation coverage

100k original rows, 1k explanation sample, three trials/backend, three rounds. The original Polars frame and model training are outside the timed region; the pandas route includes to_pandas, while Polars clones its native frame. Workflow time includes conversion, Explainer initialization and explanation. Explanation-only time excludes conversion/initialization. Each measured worker performs an excluded warm explanation first. Native pandas callbacks retain the historical NumPy-acceptance probe during initialization. RSS includes interpreter, dependencies, model, original frame and common synthetic-value factory buffers.

| Model | Family | Workflow ratio | Explanation-only ratio | pandas / Polars explanation seconds | pandas / Polars peak RSS MiB | Max parity error |
|---|---|---:|---:|---:|---:|---:|
| sklearn | importance | 1.195 | 1.112 | 0.0542 / 0.0487 | 240.3 / 234.7 | 0 |
| sklearn | breakdown | 1.049 | 0.906 | 0.0244 / 0.0269 | 240.7 / 236.2 | 0 |
| sklearn | shap | 1.023 | 0.935 | 0.0404 / 0.0432 | 241.6 / 236.7 | 0 |
| sklearn | partial | 1.078 | 0.929 | 0.0237 / 0.0255 | 245.4 / 243.9 | 0 |
| sklearn | accumulated | 1.018 | 0.981 | 0.1045 / 0.1065 | 255.1 / 254.6 | 0 |
| sklearn | conditional | 1.109 | 0.969 | 0.0306 / 0.0316 | 249.3 / 245.4 | 0 |
| lightgbm | importance | 1.152 | 1.069 | 0.0495 / 0.0463 | 241.7 / 236.6 | 0 |
| lightgbm | breakdown | 1.135 | 0.952 | 0.0208 / 0.0218 | 241.1 / 237.9 | 0 |
| lightgbm | shap | 1.066 | 0.952 | 0.0331 / 0.0348 | 242.1 / 238.8 | 0 |
| lightgbm | partial | 1.129 | 0.964 | 0.0236 / 0.0245 | 243.8 / 244.2 | 0 |
| lightgbm | accumulated | 1.046 | 1.006 | 0.1073 / 0.1067 | 255.0 / 255.0 | 0 |
| lightgbm | conditional | 1.097 | 0.952 | 0.0298 / 0.0313 | 252.1 / 248.9 | 0 |

Several explanation-only ratios regress even where workflow time benefits from avoiding input conversion. Preserve both measures; these small differences vary across runs. Before/after snapshots are both retained and do not establish a general causal speedup for the metadata change. RSS sampling every 2ms can miss peaks; allocator warming affects baselines. Cold regressions remain in earlier phase reports and CI cold/warm artifacts. No universal model acceleration or free-threading speedup is claimed.
