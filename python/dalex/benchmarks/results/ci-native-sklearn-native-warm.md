# Measured pandas versus Polars benchmark

Model path: sklearn-native. Training input: polars.

Runtime: warm in each worker. Python 3.14.8. Ratios above 1 favor Polars; below 1 favor pandas.

| Rows | pandas E2E s (IQR) | Polars E2E s (IQR) | E2E ratio | Adapter ratio | pandas/Polars peak RSS MiB | Max parity error |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.0705 (0.0704-0.0709) | 0.0570 (0.0565-0.0571) | 1.236 | 1.914 | 251.3/275.4 | 0 |
| 10000 | 0.3257 (0.3249-0.3264) | 0.3072 (0.3061-0.3080) | 1.060 | 2.366 | 256.0/280.0 | 0 |

Native frame construction and model conversion/inference are included. Training and process startup are excluded.
One fitted estimator, identical seeds, data, weights and settings per size; alternating backend order; one excluded warm-up per backend.
RSS is sampled every 2 ms in isolated processes and is a lower bound on true peak. No free-threading speed claim is made.
JSON includes every trial, inference time, adapter time, absolute and incremental RSS, dispersion, and package versions.
