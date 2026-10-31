# Measured pandas versus Polars benchmark

Model path: sklearn-numpy. Training input: numpy.

Runtime: cold in each fresh worker. Python 3.14.8. Ratios above 1 favor Polars; below 1 favor pandas.

| Rows | pandas E2E s (IQR) | Polars E2E s (IQR) | E2E ratio | Adapter ratio | pandas/Polars peak RSS MiB | Max parity error |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.0669 (0.0663-0.0669) | 0.0722 (0.0716-0.0726) | 0.926 | 1.900 | 250.8/276.8 | 0 |
| 10000 | 0.3278 (0.3265-0.3288) | 0.3294 (0.3273-0.3297) | 0.995 | 2.686 | 255.0/280.0 | 0 |

Native frame construction and model conversion/inference are included. Training and process startup are excluded.
One fitted estimator, identical seeds, data, weights and settings per size; alternating backend order; one excluded warm-up per backend.
RSS is sampled every 2 ms in isolated processes and is a lower bound on true peak. No free-threading speed claim is made.
JSON includes every trial, inference time, adapter time, absolute and incremental RSS, dispersion, and package versions.
