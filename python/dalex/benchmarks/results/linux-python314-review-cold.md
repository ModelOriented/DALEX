# Measured pandas versus Polars benchmark

Runtime: cold in each fresh worker. Python 3.14.8. Ratios above 1 favor Polars; below 1 favor pandas.

| Rows | pandas E2E s (IQR) | Polars E2E s (IQR) | E2E ratio | Adapter ratio | pandas/Polars peak RSS MiB | Max parity error |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.0657 (0.0652-0.0660) | 0.0705 (0.0700-0.0718) | 0.932 | 1.845 | 250.7/275.1 | 0 |
| 10000 | 0.3276 (0.3273-0.3281) | 0.3253 (0.3244-0.3279) | 1.007 | 2.458 | 255.1/280.0 | 0 |

Native frame construction and model conversion/inference are included. Training and process startup are excluded.
One fitted estimator, identical seeds, data, weights and settings per size; alternating backend order; one excluded warm-up per backend.
RSS is sampled every 2 ms in isolated processes and is a lower bound on true peak. No free-threading speed claim is made.
JSON includes every trial, inference time, adapter time, absolute and incremental RSS, dispersion, and package versions.
