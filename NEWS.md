DALEX 0.2.1
----------------------------------------------------------------
* The `explain()` function converts `tibbles` to `data.frame` when specified as `data` argument ([#15](https://github.com/pbiecek/DALEX/issues/15))
* The default generic `explain.default()` should help when `explain()` from `dplyr` is loaded after `DALEX` ([#16](https://github.com/pbiecek/DALEX/issues/16))

DALEX 0.2.0
----------------------------------------------------------------
* New names for some functions: `model_performance()`, `variable_importance()`, `variable_response()`, `outlier_detection()`, `prediction_breakdown()`. Old names are now deprecated but still working. ([#12](https://github.com/pbiecek/DALEX/issues/12))
* A new dataset `apartments` - will be used in examples
* `variable_importance()` allows work on full dataset if `n_sample` is negative
* `plot_model_performance()` uses ecdf or boxplots (depending on `geom` parameter).

DALEX 0.1.8
----------------------------------------------------------------
* Function `single_variable()` supports factor variables as well (with the use of `factorMerger` package). Remember to use `type='factor'` when playing with factors. ([#10](https://github.com/pbiecek/DALEX/issues/10))
* Change in the function `explain()`. Old version has an argument `predict.function`, now it's `predict_function`. New name is more consistent with other arguments. ([#7](https://github.com/pbiecek/DALEX/issues/7))
* New vigniette for `xgboost` model ([#11](https://github.com/pbiecek/DALEX/issues/11))

DALEX 0.1.1
----------------------------------------------------------------
* Support for global model structure explainers with `variable_dropout()` function 

DALEX 0.1
----------------------------------------------------------------
* DALEX package is now public
* `explain()` function implemented
* `single_prediction()` function implemented
* `single_variable()` function implemented
