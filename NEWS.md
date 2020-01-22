DALEX 1.0
----------------------------------------------------------------
* `variable_profile`  calls `ingredients::ceteris_paribus` ([#131](https://github.com/ModelOriented/DALEX/issues/131)).
* `variable_response` and `feature_response` moved to `variable_effect` and now it calls `ingredients::partial_dependency` ([#131](https://github.com/ModelOriented/DALEX/issues/131)).
* `prediction_breakdown` moved to `variable_attribution` and now it calls `iBreakDown::break_down` ([#131](https://github.com/ModelOriented/DALEX/issues/131)).
* updated `variable_importance`, not it calls the `ingredients::variable_importance` ([#131](https://github.com/ModelOriented/DALEX/issues/131)).
* updated `model_performance`  ([#130](https://github.com/ModelOriented/DALEX/issues/130)).
* added `yhat` for `lrm` models from `rms` package
* `theme_drwhy` has now left aligned title and subtitle.

DALEX 0.4.9
----------------------------------------------------------------
* updated `titanic_imputed` ([#113](https://github.com/ModelOriented/DALEX/issues/113)).
* added `weights` to the explainer. Note that not all explanations know how to handle weights ([#118](https://github.com/ModelOriented/DALEX/issues/118)).
* `yhat()` and `model_info()` now support models created with `gbm` package.

DALEX 0.4.8
----------------------------------------------------------------
* new argument `colorize` in the `explain()` as requested in ([#112](https://github.com/ModelOriented/DALEX/issues/112)).
* new generic function `model_info()`. It will extract basic irnformation like model package nam version and task type. ([#109](https://github.com/ModelOriented/DALEX/issues/109), [#110](https://github.com/ModelOriented/DALEX/issues/110))
* new functions `update_data()` and `update_label()`. ([#114](https://github.com/ModelOriented/DALEX/issues/114)))

DALEX 0.4.7
----------------------------------------------------------------
* new dataset `titanic_imputed` as requested in ([#104](https://github.com/ModelOriented/DALEX/issues/104)).
* the `explain()` function now detects if target variable `y` is present in the `data` as requested in  ([#103](https://github.com/ModelOriented/DALEX/issues/103)).
* the DALEX GitHub repository is transfered from `pbiecek/DALEX` to [ModelOriented/DALEX](https://github.com/ModelOriented/DALEX/).

DALEX 0.4.6
----------------------------------------------------------------
* Examples updated. Now they use only datasets available from DALEX.
* yhat.H2ORegressionModel and yhat.H2OBinomialModel moved to ([DALEXtra](https://github.com/ModelOriented/DALEXtra)) and merged into explain_h2o() function.
* yhat.WrappedModelmoved to ([DALEXtra](https://github.com/ModelOriented/DALEXtra)) and merged as explain_mlr() function.
* Wrapper for scikit-learn models restored in ([DALEXtra](https://github.com/ModelOriented/DALEXtra)) package.
* loss_one_minus_auc function added to loss_functions.R. It uses 1-auc to compute loss. Function created by Alicja Gosiewska.
* Extension for DALEX avaiable at ([DALEXtra](https://github.com/ModelOriented/DALEXtra))

DALEX 0.4.5
----------------------------------------------------------------
* the `explain()` function is more verbose. With `verbose = TRUE` (default) it prints detailed information about elements of an explainer ([#95](https://github.com/pbiecek/DALEX/issues/95)).

DALEX 0.4.4
----------------------------------------------------------------
* new color schemes: `colors_breakdown_drwhy()`, `colors_discrete_drwhy()` and `colors_diverging_drwhy()`.
* in this version the `scikitlearn_model()` is removed as it is not working with python 2.7

DALEX 0.4.3
----------------------------------------------------------------
* New support for scikit-learn models via `scikitlearn_model()`

DALEX 0.4.2
----------------------------------------------------------------
* New `yhat` functions for `mlr`, `h2o` and `caret` packages (added by Szymon).

DALEX 0.4.1
----------------------------------------------------------------
* `plot.variable_importance_explainer()` has now  `desc_sorting` argument. If FALSE then variable importance will be sorted in an increasing order ([#41](https://github.com/pbiecek/DALEX/issues/41)).

DALEX 0.4.0
----------------------------------------------------------------
* `ingredients` and `iBreakDown` are added to additional features ([#72](https://github.com/pbiecek/DALEX/issues/72)).
* `feature_response()` and `variable_response()` are marked as Deprecated. It is suggested to use `ingredients::partial_dependency()`, `ingredients::accumulated_dependency()` instead ([#74](https://github.com/pbiecek/DALEX/issues/74)).
* `variable_importance()` is marked as Deprecated. It is suggested to use `ingredients::feature_importance()` instead  ([#75](https://github.com/pbiecek/DALEX/issues/75)).
* `prediction_breakdown()` is marked as Deprecated. It is suggested to use `iBreakDown::break_down()` or `iBreakDown::shap()` instead  ([#76](https://github.com/pbiecek/DALEX/issues/76)).

DALEX 0.3.1
----------------------------------------------------------------
* updated filenames

DALEX 0.3
----------------------------------------------------------------
* `pdp`, `factorMerger` and `ALEPlot` are going to `Suggested`. ([#60](https://github.com/pbiecek/DALEX/issues/60)). In next releases they will be deprecated.
* added `predict` function that calls the `predict_function` hidden in the `explainer` object. ([#58](https://github.com/pbiecek/DALEX/issues/58)).

DALEX 0.2.9
----------------------------------------------------------------
* the `titanic` dataset is copied from `stablelearner` package. Some features are transformed (some `NA` replaced with `0`, more numeric features).

DALEX 0.2.8
----------------------------------------------------------------
* `DALEX` is being prepared for tighter integration with `iBreakDown` and `ingredients`.
* temporally there is a duplicated `single_variable` and `single_feature`
* Added new `theme_drwhy()`.
* New arguments in the `plot.variable_importance_explainer()`. Namely `bar_width` with widths of bars and `show_baseline` if baseline shall be included in these plots.
* New skin in the `plot.variable_response_explainer()`.
* New skin in the `plot.prediction_breakdown_explainer()`.


DALEX 0.2.7
----------------------------------------------------------------
* Test datasets are now named `apartments_test` and `HR_test`
* For binary classification we return just a second column. NOTE: this may cause some unexpected problems with code dependend on defaults for DALEX 0.2.6.

DALEX 0.2.6
----------------------------------------------------------------
* New versions of `yhat` for `ranger` and `svm` models.

DALEX 0.2.5
----------------------------------------------------------------
* Residual distribution plots for model performance are now more legible when multiple models are plotted. The styling of plot and axis titles have also been improved (@kevinykuo).
* The defaults of `single_prediction()` are now consistent with `breakDown::broken()`. Specifically, `baseline` is now `0` by default instead of `"Intercept"`. The user can also specify the `baseline` and other arguments by passing them to `single_prediction` (@kevinykuo, [#39](https://github.com/pbiecek/DALEX/issues/39)). **WARNING:** Change in the default value of `baseline`.
* New `yhat.*` functions help to handle additional parameters to different `predict()` functions.
* Updated `CITATION` info


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
