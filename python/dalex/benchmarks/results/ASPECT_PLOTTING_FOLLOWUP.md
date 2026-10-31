# Aspect plotting checkpoint: d9f2e55

[Implementation](https://github.com/brunocarlin/DALEX/commit/d9f2e55230b9b307b507b4e3093e284c511eda2f) moves Model/Predict Aspect Importance and both triplots to native numeric tables with separate heterogeneous metadata. Writable pandas views remain authoritative. Native hover metadata preserves large integers, nanosecond/timezone values and fit-time observation snapshots. Native widget callbacks retain explanation results without publishing pandas tables.

## Evidence for this checkpoint

- 85 focused cases pass on primary, minimum Polars/Narwhals, pandas 3.1 and pandas nightly stacks.
- Primary local full suite: **1,716 passed / 18 optional PPS skips** before whitespace cleanup. The exact committed source also passes the complete pandas-nightly suite with the same counts.
- [All 21 exact-commit fork CI jobs passed](https://github.com/brunocarlin/DALEX/actions/runs/41164237074). Artifact audit verifies 15 full suites (1,716/18 each), eight PyTorch cases, eight TensorFlow cases and 18 PPS cases in their compatible environment. Actual 3.14t GIL state is disabled after 21 dependency imports; unforced SHAP import enables it.
- Independent upstream comparison covers **174 figures**: 168 complete payloads and six triplots after removing only redundant unlabelled collinear hover samples. Vertices, labelled points, hover text and layout match. Browser-resolved padding previously determined hover-point density; explicit finite spans now determine it. That density difference is an intentional compatibility exception, not exact payload equality.
- The existing 27 numerical reference outputs match (CI maximum absolute error 8.88e-16); 15 actual upstream pickle objects load. These tests do not establish every possible numerical contract.
- All six current CI sklearn/LightGBM importance benchmark reports match the committed library fingerprint and have zero backend parity error. No separate Aspect plotting speedup is claimed.

Source identity: 135 modules; SHA256 `8f316bc933758537f5f68e112a9a283ce17e66bd0a4fd41bcdad944520af31ed`. [CI audit](native-aspect-plotting-followup/ci-validation.json), [upstream figures](native-aspect-plotting-followup/upstream-plots.json), [artifact fingerprints](native-aspect-plotting-followup/manifest.json).

## A limitation found by stronger subsequent validation

The earlier numerical probes used linear models for hierarchical Aspect checks. A later six-feature nonlinear example found that **d9f2e55 resets the serial Shapley path RNG at every cut**, producing a maximum importance difference of **0.4574545767390757** from untouched upstream. Linear models concealed the error because their path contributions do not depend on order. This checkpoint is therefore **not complete hierarchical-Shapley compatibility evidence**.

The subsequent shared-kernel consolidation fixes the stream to advance across cuts and the final single-variable calculation, and adds a fixed upstream regression case on pandas/Polars/Arrow plus two new nonlinear numerical comparisons. Its validation must be read separately; green d9f2e55 checks do not certify that later source. Index-sensitive pandas Aspect callback compatibility also receives stronger coverage in that follow-up.

CP and fairness plotting have separate source-stamped reports. Older private algorithms, fairness text-check preparation and pandas performance regressions still require work at this checkpoint. The PR remains draft and unmerged.
