# Measured pandas versus Polars benchmark

Model path: sklearn-numpy. Training input: numpy.

Runtime: warm in each worker. Python 3.14.8. Ratios above 1 favor Polars; below 1 favor pandas.

| Rows | pandas E2E s (IQR) | Polars E2E s (IQR) | E2E ratio | Adapter ratio | pandas/Polars peak RSS MiB | Max parity error |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.0616 (0.0613-0.0617) | 0.0556 (0.0555-0.0556) | 1.107 | 1.855 | 250.7/277.2 | 0 |
| 10000 | 0.3220 (0.3216-0.3236) | 0.3093 (0.3086-0.3113) | 1.041 | 2.652 | 255.2/280.2 | 0 |

Native frame construction and model conversion/inference are included. Training and process startup are excluded.
One fitted estimator, identical seeds, data, weights and settings per size; alternating backend order; one excluded warm-up per backend.
RSS is sampled every 2 ms in isolated processes and is a lower bound on true peak. No free-threading speed claim is made.
JSON includes every trial, inference time, adapter time, absolute and incremental RSS, dispersion, and package versions.
