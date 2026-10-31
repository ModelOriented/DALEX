# Measured pandas versus Polars benchmark

Model path: lightgbm-native. Training input: polars.

Runtime: cold in each fresh worker. Python 3.14.8. Ratios above 1 favor Polars; below 1 favor pandas.

| Rows | pandas E2E s (IQR) | Polars E2E s (IQR) | E2E ratio | Adapter ratio | pandas/Polars peak RSS MiB | Max parity error |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.0714 (0.0712-0.0717) | 0.0688 (0.0687-0.0700) | 1.038 | 1.882 | 255.7/280.5 | 0 |
| 10000 | 0.3557 (0.3545-0.3560) | 0.3550 (0.3524-0.3653) | 1.002 | 2.696 | 259.6/285.2 | 0 |

Native frame construction and model conversion/inference are included. Training and process startup are excluded.
One fitted estimator, identical seeds, data, weights and settings per size; alternating backend order; one excluded warm-up per backend.
RSS is sampled every 2 ms in isolated processes and is a lower bound on true peak. No free-threading speed claim is made.
JSON includes every trial, inference time, adapter time, absolute and incremental RSS, dispersion, and package versions.
