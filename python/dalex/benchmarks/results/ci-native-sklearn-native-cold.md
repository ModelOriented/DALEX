# Measured pandas versus Polars benchmark

Model path: sklearn-native. Training input: polars.

Runtime: cold in each fresh worker. Python 3.14.8. Ratios above 1 favor Polars; below 1 favor pandas.

| Rows | pandas E2E s (IQR) | Polars E2E s (IQR) | E2E ratio | Adapter ratio | pandas/Polars peak RSS MiB | Max parity error |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.0847 (0.0846-0.0848) | 0.0700 (0.0698-0.0704) | 1.209 | 1.888 | 251.2/277.3 | 0 |
| 10000 | 0.3456 (0.3430-0.3469) | 0.3195 (0.3194-0.3203) | 1.082 | 2.294 | 256.1/280.0 | 0 |

Native frame construction and model conversion/inference are included. Training and process startup are excluded.
One fitted estimator, identical seeds, data, weights and settings per size; alternating backend order; one excluded warm-up per backend.
RSS is sampled every 2 ms in isolated processes and is a lower bound on true peak. No free-threading speed claim is made.
JSON includes every trial, inference time, adapter time, absolute and incremental RSS, dispersion, and package versions.
