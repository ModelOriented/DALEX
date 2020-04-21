dalex (development)
----------------------------------------------------------------
* ...

dalex 0.1.5
----------------------------------------------------------------
* *feature*: `xgboost` package compatibility
* *feature*: added `model_class` parameter to `Explainer` to handle wrapped models
* *feature*: `Exaplainer`s `model_info` remembers if parameters are default
* *bugfix:* `variable_groups` parameter now works correctly in `model_parts`
* *constructor*: changed parameter order in `Explainer`: `model_type`, `model_info`, `colorize`
* *documentation:* `model_parts` documentation is updated

dalex 0.1.4
----------------------------------------------------------------
* *feature:* new `Explainer.residual` method which uses
`residual_function` to calculate `residuals`
* *feature:* new `dump` and `dumps` methods for saving `Explainer` in a binary form;
`load` and `loads` methods for loading `Explainer` from binary form
* *fix:* `Explainer` constructor verbose text
* *bugfix:* `B:=B+1` - `Shap` now stores average results as `B=0` and path results as `B=1,2,...`
* *bugfix:* `Explainer.model_performance` method uses `self.model_type` when
 `model_type` is `None`
* *bugfix:* values in `BreakDown` and `Shap` are now rounded to 4 significant
 places (#180)
* *bugfix:* `Shap` by default uses `path='average'`, `sign` column is
properly updated and bars in `plot` are sorted by `abs(contribution)`

dalex 0.1.3
----------------------------------------------------------------
* [release](https://medium.com/@ModelOriented/xai-in-python-with-dalex-4b173486aa92) of the `dalex` package
* `Explainer` object with `predict`, `predict_parts`, `predict_profile`,
`model_performance`, `model_parts` and `model_profile` methods
* `BreakDown`, `Shap`, `CeterisParibus`, `ModelPerformance`,
`VariableImportance` and `AggregatedProfiles` objects with a `plot` method
* `load_titanic()` function which loads the `titanic_imputed` dataset
