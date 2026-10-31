# Measured pandas versus Polars benchmark

Runtime: warm in each worker. Python 3.14.9. Ratios above 1 favor Polars; below 1 favor pandas.

| Rows | pandas E2E s (IQR) | Polars E2E s (IQR) | E2E ratio | Adapter ratio | pandas/Polars peak RSS MiB | Max parity error |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.0503 (0.0496-0.0507) | 0.0451 (0.0441-0.0459) | 1.116 | 1.803 | 292.0/312.9 | 0 |
| 10000 | 0.2544 (0.2540-0.2554) | 0.2450 (0.2446-0.2591) | 1.038 | 2.619 | 296.2/317.2 | 0 |

Native frame construction and model conversion/inference are included. Training and process startup are excluded.
One fitted estimator, identical seeds, data, weights and settings per size; alternating backend order; one excluded warm-up per backend.
RSS is sampled every 2 ms in isolated processes and is a lower bound on true peak. No free-threading speed claim is made.
JSON includes every trial, inference time, adapter time, absolute and incremental RSS, dispersion, and package versions.
