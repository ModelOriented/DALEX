# Measured pandas versus Polars benchmark

Runtime: warm in each worker. Python 3.14.8. Ratios above 1 favor Polars; below 1 favor pandas.

| Rows | pandas E2E s (IQR) | Polars E2E s (IQR) | E2E ratio | Adapter ratio | pandas/Polars peak RSS MiB | Max parity error |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.0600 (0.0598-0.0618) | 0.0560 (0.0555-0.0561) | 1.071 | 1.879 | 250.5/276.9 | 0 |
| 10000 | 0.3231 (0.3218-0.3232) | 0.3113 (0.3089-0.3134) | 1.038 | 2.406 | 255.2/279.9 | 0 |

Native frame construction and model conversion/inference are included. Training and process startup are excluded.
One fitted estimator, identical seeds, data, weights and settings per size; alternating backend order; one excluded warm-up per backend.
RSS is sampled every 2 ms in isolated processes and is a lower bound on true peak. No free-threading speed claim is made.
JSON includes every trial, inference time, adapter time, absolute and incremental RSS, dispersion, and package versions.
