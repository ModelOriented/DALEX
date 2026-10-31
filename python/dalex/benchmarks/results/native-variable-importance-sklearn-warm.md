# Measured pandas versus Polars benchmark

Model path: sklearn-native. Training input: polars.

Runtime: warm in each worker. Python 3.14.4. Ratios above 1 favor Polars; below 1 favor pandas.

| Rows | pandas E2E s (IQR) | Polars E2E s (IQR) | E2E ratio | Adapter ratio | pandas/Polars peak RSS MiB | Max parity error |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.0513 (0.0509-0.0521) | 0.0433 (0.0429-0.0446) | 1.184 | 1.551 | 214.6/217.8 | 0 |
| 10000 | 0.3059 (0.3033-0.3111) | 0.2998 (0.2976-0.3018) | 1.021 | 2.348 | 220.1/225.5 | 0 |
| 100000 | 2.8325 (2.8299-2.8394) | 2.7764 (2.7728-2.8059) | 1.020 | 6.620 | 267.0/269.4 | 0 |

Native frame construction and model conversion/inference are included. Training and process startup are excluded.
One fitted estimator, identical seeds, data, weights and settings per size; alternating backend order; one excluded warm-up per backend.
RSS is sampled every 2 ms in isolated processes and is a lower bound on true peak. No free-threading speed claim is made.
JSON includes every trial, inference time, adapter time, absolute and incremental RSS, dispersion, and package versions.
