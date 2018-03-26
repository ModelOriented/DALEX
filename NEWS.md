DALEX 0.2.0
----------------------------------------------------------------
* New names for some functions: `model_performance()`, `variable_importance()`, `variable_response()`, `outlier_detection()`, `prediction_breakdown()`. Old names are now deprecated but still working. ([#12](https://github.com/pbiecek/archivist/issues/12))

DALEX 0.1.8
----------------------------------------------------------------
* Function `single_variable()` supports factor variables as well (with the use of `factorMerger` package). Remember to use `type='factor'` when playing with factors. ([#10](https://github.com/pbiecek/archivist/issues/10))
* Change in the function `explain()`. Old version has an argument `predict.function`, now it's `predict_function`. New name is more consistent with other arguments. ([#7](https://github.com/pbiecek/archivist/issues/7))
* New vigniette for `xgboost` model ([#11](https://github.com/pbiecek/archivist/issues/11))

DALEX 0.1.1
----------------------------------------------------------------
* Support for global model structure explainers with `variable_dropout()` function 

DALEX 0.1
----------------------------------------------------------------
* DALEX package is now public
* `explain()` function implemented
* `single_prediction()` function implemented
* `single_variable()` function implemented
