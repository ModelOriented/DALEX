# Measured pandas versus Polars benchmark

Runtime: warm in each worker. Python 3.14.4. Ratios above 1 favor Polars; below 1 favor pandas.

| Rows | pandas E2E s (IQR) | Polars E2E s (IQR) | E2E ratio | Adapter ratio | pandas/Polars peak RSS MiB | Max parity error |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.0463 (0.0444-0.0466) | 0.0442 (0.0441-0.0455) | 1.047 | 1.400 | 214.3/217.8 | 0 |
| 10000 | 0.2955 (0.2943-0.2973) | 0.2998 (0.2960-0.3004) | 0.986 | 2.351 | 218.4/225.7 | 0 |
| 100000 | 0.2967 (0.2964-0.2973) | 0.3054 (0.3032-0.3057) | 0.971 | 1.524 | 233.6/251.7 | 0 |

Native frame construction and model conversion/inference are included. Training and process startup are excluded.
One fitted estimator, identical seeds, data, weights and settings per size; alternating backend order; one excluded warm-up per backend.
RSS is sampled every 2 ms in isolated processes and is a lower bound on true peak. No free-threading speed claim is made.
JSON includes every trial, inference time, adapter time, absolute and incremental RSS, dispersion, and package versions.

Measured source revision: `4eac67037ca15af8701ba1a6c37bf13e4510b344`.
