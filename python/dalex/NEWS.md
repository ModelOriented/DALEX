dalex (development)
----------------------------------------------------------------
* *fix:* `vertical_spacing` in `VariableImportance.plot` when `split='variable'`
* *feature:* added `disable` parameter where `tqdm` is used to verbose progress bar
* *feature:* added `loss_one_minus_auc` function that can be used with `loss_function='1-auc'` in `model_parts`
* *defaults:* use different `loss_function` for classification and regression ([#248](https://github.com/ModelOriented/DALEX/issues/248))
* *defaults:* models that use `proba` yhats now get `model_type='classification'` if it's not specified
* *fix:* `loss_function='auc'` now uses `loss_one_minus_auc` as this should be a descending measure

dalex 0.1.9
----------------------------------------------------------------
* *feature:* added `random_state` parameter for `SHAP` and `model_profile` for reproducible calculations
* *fix:* fixed `random_state` parameter in `model_parts`
* *feature:* multiprocessing added for: `model_profile`, `model_parts`, `predict_profile` and `predict_parts(type='shap')`, through the `processes` parameter
* *fix:* significantly improved the speed of `accumulated` and `conditional` types in `model_profile`
* *bugfix:* use [pd.api.types.is_numeric_dtype()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.types.is_numeric_dtype.html)
  instead of `np.issubdtype()` to cover more types; e.g. it caused errors with `string` type
* *defaults:* use [pd.convert_dtypes()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.convert_dtypes.html)
 on the result of `CeterisParibus` to fix variable dtypes and
 later allow for a concatenation without the dtype conversion
* *fix:* `variables` parameter now can be a single `str` value
* *fix:* number rounding in `predict_parts`, `model_parts` ([#245](https://github.com/ModelOriented/DALEX/issues/245))
* *fix:* CP calculations for models that take only variables as an input 

dalex 0.1.8
----------------------------------------------------------------
* *bugfix:* `variable_splits` parameter now works correctly in `predict_profile`
* *bugfix:* fix baseline for 3+ models in `AggregatedProfiles.plot` ([#234](https://github.com/ModelOriented/DALEX/issues/234))
* *printing:* now rounding numbers in `Explainer` messages 
* *fix:* minor checks fixes in `instance_level`
* *bugfix:* `AggregatedProfiles.plot` now works with `groups`

dalex 0.1.7
----------------------------------------------------------------
* *feature:* parameter `N` in `model_profile` can be set to `None`, to select all observations 
* *input:* `groups` and `variable` parameters in `model_profile` can be: `str`, `list`, `numpy.ndarray`, `pandas.Series`
* *fix:* `check_label` returned only a first letter
* *bugfix:* removed the conversion of `all_variables` to `str` in
`prepare_all_variables`, which caused an error in `model_profile` ([#214](https://github.com/ModelOriented/DALEX/issues/214))
* *defaults:* change numpy data variable names from numbers to strings

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
