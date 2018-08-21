DALEX 0.2.5
----------------------------------------------------------------
* The defaults of `single_prediction()` are now consistent with `breakDown::broken()`. Specifically, `baseline` is now `0` by default instead of `"Intercept"`. The user can also specify the `baseline` and other arguments by passing them to `single_prediction`. ([#39](https://github.com/pbiecek/DALEX/issues/39))


DALEX 0.2.4
----------------------------------------------------------------
* New dataset `HR` and `HRTest`. Target variable is a factor with three levels. Is used in examples for classification.
* The `plot.model_performance()` has now `show_outliers` parameter. Set it to anything >0 and observations with largest residuals will be presented in the plot. ([#34](https://github.com/pbiecek/DALEX/issues/34))

DALEX 0.2.3
----------------------------------------------------------------
* Small fixes in `variable_response()` to better support of `gbm` models (c8393120ffb05e2f3c70b0143c4e92dc91f6c823).
* Better title for `plot_model_performance()` (e5e61d0398459b78ea38ccc980c4040fd853f449).
* Tested with `breakDown` v 0.1.6.

DALEX 0.2.2
----------------------------------------------------------------
* The `single_variable() / variable_response()` function uses `predict_function` from `explainer` ([#17](https://github.com/pbiecek/DALEX/issues/17))

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
