# Measured pandas versus Polars benchmark

Model path: lightgbm-native. Training input: polars.

Runtime: warm in each worker. Python 3.14.8. Ratios above 1 favor Polars; below 1 favor pandas.

| Rows | pandas E2E s (IQR) | Polars E2E s (IQR) | E2E ratio | Adapter ratio | pandas/Polars peak RSS MiB | Max parity error |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.0661 (0.0661-0.0663) | 0.0562 (0.0562-0.0572) | 1.176 | 1.774 | 255.7/280.9 | 0 |
| 10000 | 0.3508 (0.3491-0.3510) | 0.3394 (0.3377-0.3394) | 1.034 | 2.628 | 259.9/285.2 | 0 |

Native frame construction and model conversion/inference are included. Training and process startup are excluded.
One fitted estimator, identical seeds, data, weights and settings per size; alternating backend order; one excluded warm-up per backend.
RSS is sampled every 2 ms in isolated processes and is a lower bound on true peak. No free-threading speed claim is made.
JSON includes every trial, inference time, adapter time, absolute and incremental RSS, dispersion, and package versions.
