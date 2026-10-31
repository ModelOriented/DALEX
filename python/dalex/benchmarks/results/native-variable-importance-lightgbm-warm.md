# Measured pandas versus Polars benchmark

Model path: lightgbm-native. Training input: polars.

Runtime: warm in each worker. Python 3.14.4. Ratios above 1 favor Polars; below 1 favor pandas.

| Rows | pandas E2E s (IQR) | Polars E2E s (IQR) | E2E ratio | Adapter ratio | pandas/Polars peak RSS MiB | Max parity error |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.0477 (0.0476-0.0484) | 0.0420 (0.0411-0.0420) | 1.136 | 2.066 | 215.9/220.0 | 0 |
| 10000 | 0.2671 (0.2662-0.2700) | 0.2611 (0.2593-0.2631) | 1.023 | 2.193 | 220.4/226.3 | 0 |
| 100000 | 2.5441 (2.5391-2.5522) | 2.4170 (2.4085-2.4372) | 1.053 | 7.056 | 267.7/264.7 | 0 |

Native frame construction and model conversion/inference are included. Training and process startup are excluded.
One fitted estimator, identical seeds, data, weights and settings per size; alternating backend order; one excluded warm-up per backend.
RSS is sampled every 2 ms in isolated processes and is a lower bound on true peak. No free-threading speed claim is made.
JSON includes every trial, inference time, adapter time, absolute and incremental RSS, dispersion, and package versions.
