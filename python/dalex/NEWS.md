dalex (development)
----------------------------------------------------------------
* ...

dalex 0.1.6
----------------------------------------------------------------
* *fix:* change `short_name` encoding in `fifa` dataset (utf8->ascii)
* *fix:* remove `scipy` dependency
* *defaults:* default `loss_root_mean_square` in model parts changed to `rmse`
* *bugfix:* checks related to `new_observation` in `BreakDown, Shap,
CeterisParibus` now work for multiple inputs
([#207](https://github.com/ModelOriented/DALEX/issues/207))
* *bugfix:* `CeterisParibus.fit` and `CeterisParibus.plot` now work for
more types of `new_observation.index`, but won't work for a `bolean` type
([#211](https://github.com/ModelOriented/DALEX/issues/211))

dalex 0.1.5
----------------------------------------------------------------
* *feature:* add `xgboost` package compatibility ([#188](https://github.com/ModelOriented/DALEX/issues/188))
* *feature:* added `model_class` parameter to `Explainer` to handle wrapped
models
* *feature:* `Exaplainer`s `model_info` remembers if parameters are default
* *bugfix:* `variable_groups` parameter now works correctly in `model_parts`
* *fix:* changed parameter order in `Explainer`: `model_type`, `model_info`,
`colorize`
* *documentation:* `model_parts` documentation is updated
* *feature:* new `show` parameter in `plot` methods that (`if False`) returns
`plotly Figure` ([#190](https://github.com/ModelOriented/DALEX/issues/190))
* *feature:* `load_fifa()` function which loads the preprocessed [players_20
dataset](https://www.kaggle.com/stefanoleone992/fifa-20-complete-player-dataset)
* *fix:* `CeterisParibus.plot` tooltip

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
 places ([#180](https://github.com/ModelOriented/DALEX/issues/180))
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
