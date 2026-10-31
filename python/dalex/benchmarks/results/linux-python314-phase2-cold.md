# Measured pandas versus Polars benchmark

Runtime: cold in each fresh worker. Python 3.14.9. Ratios above 1 favor Polars; below 1 favor pandas.

| Rows | pandas E2E s (IQR) | Polars E2E s (IQR) | E2E ratio | Adapter ratio | pandas/Polars peak RSS MiB | Max parity error |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.0509 (0.0506-0.0514) | 0.0561 (0.0552-0.0564) | 0.907 | 1.795 | 292.2/312.5 | 0 |
| 10000 | 0.2577 (0.2575-0.2587) | 0.2605 (0.2587-0.2606) | 0.990 | 2.487 | 296.2/317.5 | 0 |

Native frame construction and model conversion/inference are included. Training and process startup are excluded.
One fitted estimator, identical seeds, data, weights and settings per size; alternating backend order; one excluded warm-up per backend.
RSS is sampled every 2 ms in isolated processes and is a lower bound on true peak. No free-threading speed claim is made.
JSON includes every trial, inference time, adapter time, absolute and incremental RSS, dispersion, and package versions.
