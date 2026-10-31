# Measured pandas versus Polars benchmark

Python 3.14.8. Ratios above 1 favor Polars; below 1 favor pandas.

| Rows | pandas E2E s (IQR) | Polars E2E s (IQR) | E2E ratio | Adapter ratio | pandas/Polars peak RSS MiB | Max parity error |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.0602 (0.0602-0.0607) | 0.0712 (0.0709-0.0717) | 0.846 | 1.183 | 250.6/278.5 | 0 |
| 10000 | 0.3088 (0.3081-0.3094) | 0.3280 (0.3241-0.3317) | 0.941 | 0.978 | 255.1/283.2 | 0 |

Native frame construction and model conversion/inference are included. Training and process startup are excluded.
One fitted estimator, identical seeds, data, weights and settings per size; alternating backend order; one excluded warm-up per backend.
RSS is sampled every 2 ms in isolated processes and is a lower bound on true peak. No free-threading speed claim is made.
JSON includes every trial, inference time, adapter time, absolute and incremental RSS, dispersion, and package versions.

Measured source revision: `0cbe570e5013cdd0b931c17a36243eab5b266021`.
