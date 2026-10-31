# Measured pandas versus Polars benchmark

Python 3.14.4. Ratios above 1 favor Polars; below 1 favor pandas.

| Rows | pandas E2E s (IQR) | Polars E2E s (IQR) | E2E ratio | Adapter ratio | pandas/Polars peak RSS MiB | Max parity error |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.0557 (0.0556-0.0568) | 0.0778 (0.0771-0.0778) | 0.716 | 1.164 | 210.9/215.0 | 0 |
| 10000 | 0.3089 (0.3087-0.3120) | 0.3374 (0.3363-0.3404) | 0.916 | 0.864 | 216.0/221.7 | 0 |
| 100000 | 0.3156 (0.3126-0.3164) | 0.3409 (0.3397-0.3413) | 0.926 | 0.780 | 230.5/241.2 | 0 |

Native frame construction and model conversion/inference are included. Training and process startup are excluded.
One fitted estimator, identical seeds, data, weights and settings per size; alternating backend order; one excluded warm-up per backend.
RSS is sampled every 2 ms in isolated processes and is a lower bound on true peak. No free-threading speed claim is made.
JSON includes every trial, inference time, adapter time, absolute and incremental RSS, dispersion, and package versions.

Measured source revision: `0cbe570e5013cdd0b931c17a36243eab5b266021`.
