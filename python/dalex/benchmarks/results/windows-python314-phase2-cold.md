# Measured pandas versus Polars benchmark

Python 3.14.4. Ratios above 1 favor Polars; below 1 favor pandas.

| Rows | pandas E2E s (IQR) | Polars E2E s (IQR) | E2E ratio | Adapter ratio | pandas/Polars peak RSS MiB | Max parity error |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.0580 (0.0574-0.0601) | 0.0756 (0.0750-0.0764) | 0.767 | 2.005 | 214.1/217.7 | 0 |
| 10000 | 0.3102 (0.3090-0.3106) | 0.3289 (0.3284-0.3300) | 0.943 | 2.114 | 218.7/224.4 | 0 |
| 100000 | 0.3131 (0.3130-0.3144) | 0.3375 (0.3349-0.3398) | 0.928 | 1.628 | 233.4/243.9 | 0 |

Native frame construction and model conversion/inference are included. Training and process startup are excluded.
One fitted estimator, identical seeds, data, weights and settings per size; alternating backend order; one excluded warm-up per backend.
RSS is sampled every 2 ms in isolated processes and is a lower bound on true peak. No free-threading speed claim is made.
JSON includes every trial, inference time, adapter time, absolute and incremental RSS, dispersion, and package versions.

Measured source revision: `a1dac26144ebb234808a440fb635ceefbff255ed`.
