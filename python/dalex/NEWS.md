## Changelog

development
----------------------------------------------------------------

#### breaking changes

* method `set_options` in Arena now takies `option_category` instead of `plot_type` (`SHAPValues` => `ShapleyValues`, `FeatureImportance` => `VariableImportance`)

#### fixes

* fixed wrong error value when no `predict_function` is found in `Explainer` ([77ca90d](https://github.com/ModelOriented/DALEX/commit/77ca90d))
* set multiprocessing context to 'spawn'

#### features

* add resource mechanism to Arena
* add ShapleyValuesImportance and ShapleyValuesDependence charts to Arena

v1.1.0 (18/04/2021)
----------------------------------------------------------------

#### breaking changes

* fixed concurrent random seeds when `processes > 1` ([#392](https://github.com/ModelOriented/DALEX/issues/392)), which means that the results of parallel computation will vary between `v1.1.0` and previous versions

#### fixes

* `GroupFairnessX.plot(type='fairness_check')` generates ticks according to the x-axis range ([#409](https://github.com/ModelOriented/DALEX/issues/409))
* `GroupFainressRegression.plot(type='density')` has a more readable hover - only for outliers ([#409](https://github.com/ModelOriented/DALEX/issues/409))
* `BreakDown.plot()` wrongly displayed the "+all factors" bar when `max_vars < p` ([#401](https://github.com/ModelOriented/DALEX/issues/401))
* `GroupFairnessClassification.plot(type='metric_scores')` did not handle `NaN`'s ([#399](https://github.com/ModelOriented/DALEX/issues/399))

#### features 

* Experimental support for regression models in the `fairness` module. Added `GroupFairnessRegression` object, with the `plot` method having two types: `fairness_check` and `density`. `Explainer.model_fairness` method now depends on the `model_type` attribute. ([#391](https://github.com/ModelOriented/DALEX/issues/391))
* added `N` parameter to the `predict_parts` method which is `None` by default ([#402](https://github.com/ModelOriented/DALEX/issues/402))
* `epsilon` is now an argument of the `GroupFairnessClassification` object ([#397](https://github.com/ModelOriented/DALEX/issues/397))


v1.0.1 (19/02/2021)
----------------------------------------------------------------

#### fixes

* fixed broken range on `yaxis` in `fairness_check` plot ([#376](https://github.com/ModelOriented/DALEX/issues/376))
* warnings because `np.float` is depracated since `numpy` v1.20 ([#384](https://github.com/ModelOriented/DALEX/issues/384))

#### other 

* added `ipython` to test dependencies

v1.0.0 (29/12/2020)
----------------------------------------------------------------

#### breaking changes

These are summed up in ([#368](https://github.com/ModelOriented/DALEX/issues/368)):

* rename modules: `dataset_level` into `model_explanations`, `instance_level` into `predict_explanations`, `_arena` module into `arena` 
* use `__dir__` method to define autocompletion in IPython environment - show only `['Explainer', 'Arena', 'fairness', 'datasets']`
* add `plot` method and `result` attribute to `LimeExplanation` (use `lime.explanation.Explanation.as_pyplot_figure()` and `lime.explanation.Explanation.as_list()`)
* `CeterisParibus.plot(variable_type='categorical')` now has horizontal barplots - `horizontal_spacing=None` by default (varies on `variable_type`). Also, once again added the "dot" for observation value.
* `predict_fn` in `predict_surrogate` now uses `predict_function` (trying to make it work for more frameworks)

#### fixes

* fixed wrong verbose output when any value in `y_hat/residuals` was an `int` not `float`
* added proper `"-"` sign to negative dropout losses in `VariableImportance.plot`

#### features

* added `geom='bars'` to `AggregateProfiles.plot` to force the categorical plot 
* added `geom='roc'` and `geom='lift'` to `ModelPerformance.plot`
* added Fairness plot to Arena

#### other

* remove `colorize` from `Explainer`
* updated the documentation, refactored code (import modules not functions, unify variable names in `object.py`, move utils funcitons from `checks.py` to `utils.py`, etc.)
* added license notice next to data

v0.4.1 (02/12/2020)
----------------------------------------------------------------

* added support for `h2o.estimators.*` ([#332](https://github.com/ModelOriented/DALEX/issues/332))
* added `tensorflow.python.keras.engine.functional.Functional` to the `tensorflow` list
* updated the `plotly` dependency to `>=4.12.0`
* code maintenance: `yhat`, `check_data`

#### fixes

* fixed `check_if_empty_fields()` used in loading the `Explainer` from a pickle file, since several checks were changed
* fixed `plot()` method in `GroupFairnessClassification` as it omitted plotting a metric when `NaN` was present in metric ratios (result)
* fixed `dragons` and `HR` datasets having `,` delimeter instead of `.`, which transformed numerical columns into categorical.
* fixed representation of the `ShapWrapper` class (removed `_repr_html_` method)

#### features

* allow for `y` to be a `pandas.DataFrame` (converted) 
* allow for `data`, `y` to be a `H2OFrame` (converted)  
* added `label` parameter to all the relevant `dx.Explainer` methods, which overrides the default label in explanation's `result`
* now using `GradientExplainer` for `tf.keras.engine.sequential.Sequential`, added proper warning when `shap_explainer_type` is `None` ([#366](https://github.com/ModelOriented/DALEX/issues/366))

#### defaults

* unify verbose output of `Explainer`

v0.4.0 (17/11/2020)
----------------------------------------------------------------

* added new `arena` module, which adds the backend for Arena dashboard [@piotrpiatyszek](https://github.com/piotrpiatyszek)

#### features

* added new aliases to `dx.Explainer` methods ([#350](https://github.com/ModelOriented/DALEX/issues/350))
 in `model_parts` it is `{'permutational': 'variable_importance', 'feature_importance': 'variable_importance'}`,
 in `model_profile` it is `{'pdp': 'partial', 'ale': 'accumulated'}`
* added `Arena` object for dashboard backend. See https://github.com/ModelOriented/Arena
* new `fairness` plot types: `stacked`, `radar`, `performance_and_fairness`, `heatmap`, `ceteris_paribus_cutoff`
* upgraded `fairness_check()`

v0.3.0 (26/10/2020)
----------------------------------------------------------------

* added new `fairness` module, which will focus on bias detection, visualization and mitigation [@jakwisn](https://github.com/jakwisn)

#### fixes

* removed unnecessary warning when `precalculate=False and verbose=False` ([#340](https://github.com/ModelOriented/DALEX/issues/340))

#### features

* added `model_fairness` method to the `Explainer`, which performs fairness explanation
* added `GroupFairnessClassification` object, with the `plot` method having two types: `fairness_check` and `metric_scores`

#### defaults

* added the `N=50000` argument to `ResidualDiagnostics.plot`, which samples observations from 
 the `result` parameter to omit performance issues when `smooth=True` ([#341](https://github.com/ModelOriented/DALEX/issues/341))

dalex 0.2.2
----------------------------------------------------------------

* added support for `tensorflow.python.keras.engine.sequential.Sequential` and `tensorflow.python.keras.engine.training.Model` ([#326](https://github.com/ModelOriented/DALEX/issues/326))
* updated the `tqdm` dependency to `>=4.48.2`, `pandas` dependency to `>=1.1.2` and `numpy` dependency to `>=1.18.4`

#### fixes

* fixed the wrong order of `Explainer` verbose messages
* fixed a bug that caused `model_info` parameter to be overwritten by the default values
* fixed a bug occurring when the variable from `groups` was not of `str` type ([#327](https://github.com/ModelOriented/DALEX/issues/327))
* fixed `model_profile`: `variable_type='categorical'` not working when user passed `variables` parameter ([#329](https://github.com/ModelOriented/DALEX/issues/329)) +
 the reverse order of bars in `'categorical'` plots + (again) added `variable_splits_type` parameter to `model_profile` to specify how grid points
 shall be calculated ([#266](https://github.com/ModelOriented/DALEX/issues/266)) + allow for both `'quantile'` and `'quantiles'` types (alias)
 
#### features

* added informative error messages when importing optional dependencies ([#316](https://github.com/ModelOriented/DALEX/issues/316))
* allow for `data` and `y` to be `None` - added checks in `Explainer` methods
 
#### defaults

* wrong parameter name `title_x` changed to `y_title` in `CeterisParibus.plot` and `AggregatedProfiles.plot` ([#317](https://github.com/ModelOriented/DALEX/issues/317))
* now warning the user in `Explainer` when `predict_function` returns an error or doesn't return `numpy.ndarray (1d)` ([#325](https://github.com/ModelOriented/DALEX/issues/325))

dalex 0.2.1
----------------------------------------------------------------

* updated the `pandas` dependency to `>=1.1.0`

#### fixes

* `ModelPerformance.plot` now uses a drwhy color palette
* use `unique` method instead of `np.unique` in `variable_splits` ([#293](https://github.com/ModelOriented/DALEX/issues/293))
* `v0.2.0` didn't export new datasets
* fixed a bug where `predict_parts(type='shap')` calculated wrong `contributions` 
 ([#300](https://github.com/ModelOriented/DALEX/issues/300))
* `model_profile` uses observation mean instead of profile mean in `_yhat_` centering
* fixed barplot baseline in categorical `model_profile` and `predict_profile` plots
 ([#297](https://github.com/ModelOriented/DALEX/issues/297))
* fixed `model_profile(type='accumulated')` giving wrong results
 (#[302](https://github.com/ModelOriented/DALEX/issues/302))
* vertical/horizontal lines in plots now end on the plot edges

#### features

* added new `type='shap_wrapper'` to `predict_parts` and `model_parts` methods, which returns a new
 `ShapWrapper` object. It contains the main result attribute (`shapley_values`) and the plot method 
 (`force_plot` and `summary_plot` respectively). These come from the [shap](https://github.com/slundberg/shap) package
* `Explainer.predict` method now accepts `numpy.ndarray`
* added the `ResidualDiagnostics` object with a `plot` method
* added `model_diagnostics` method to the `Explainer`, which performs residual diagnostics
* added `predict_surrogate` method to the `Explainer`, which is a wrapper for the `lime`
 tabular explanation from the [lime](https://github.com/marcotcr/lime) package
 * added `model_surrogate` method to the `Explainer`, which creates a basic surrogate decision tree
 or linear model from the black-box model using the [scikit-learn](https://github.com/scikit-learn/scikit-learn) package
* added a `_repr_html_` method to all of the explanation objects (it prints the `result` attribute)
* added `dalex.__version__`
* added informative error messages in `Explainer` methods when `y` is of wrong type
 ([#294](https://github.com/ModelOriented/DALEX/issues/294))
* `CeterisParibus.plot(variable_type='categorical')` now allows for multiple observations
* new verbose checks for `model_type`
* add `type` to `model_info` in `dump` and `dumps` for R compatibility
 ([#303](https://github.com/ModelOriented/DALEX/issues/303))
* `ModelPerformance.result` now has `label` as index

#### defaults

* removed `_grid_` column in `AggregatedProfiles.result` and `center` only works with `type=accumulated`
* use `Pipeline._final_estimator` to extract `model_class` of the actual model
* use `model._estimator_type` to extract `model_type` if possible 

dalex 0.2.0
----------------------------------------------------------------
* major documentation update ([#270](https://github.com/ModelOriented/DALEX/issues/270))
* unified the order of function parameters

#### fixes

* `v0.1.9` had wrong `_original_` column in `predict_profile`
* `vertical_spacing` acts as intended in `VariableImportance.plot` when `split='variable'`
* `loss_function='auc'` now uses `loss_one_minus_auc` as this should be a descending measure
* plots are now saved with the original height and width
* `model_profile` now properly passes the `variables` parameter to `CeterisParibus`
* `variables` parameter in `predict_profile` now can also be a string

#### features

* use `px.express` instead of core `plotly` to make `model_profile` and `predict_profile` plots;
 thus, enhance performance and scalability
* added `verbose` parameter where `tqdm` is used to verbose progress bar
* added `loss_one_minus_auc` function that can be used with `loss_function='1-auc'` in `model_parts`
* added new example data sets: `apartments`, `dragons` and `hr`
* added `color`, `opacity`, `title_x` parameters to `model_profile` and `predict_profile` plots ([#236](https://github.com/ModelOriented/DALEX/issues/236)),
 changed tooltips and legends ([#262](https://github.com/ModelOriented/DALEX/issues/262))
* added `geom='profiles'` parameter to `model_profile` plot and `raw_profiles` attribute to `AggregatedProfiles`
* added `variable_splits_type` parameter to `predict_profile` to specify how grid points
 shall be calculated ([#266](https://github.com/ModelOriented/DALEX/issues/266))
* added `variable_splits_with_obs` parameter to `predict_profile` function to extend split points with observation
 variable values ([#269](https://github.com/ModelOriented/DALEX/issues/269))
* added `variable_splits` parameter to `model_profile`

#### defaults

* use different `loss_function` for classification and regression ([#248](https://github.com/ModelOriented/DALEX/issues/248))
* models that use `proba` yhats now get `model_type='classification'` if it's not specified
* use uniform way of grid points calculation in `predict_profile` and `model_profile` (see `variable_splits_type` parameter)
* add the variable values of `new_observation` to `variable_splits` in `predict_profile` (see `variable_splits_with_obs` parameter)
* use `N=1000` in `model_parts` and `N=300` in `model_profile` to comply with the R version
* `keep_raw_permutation` is now set to `False` instead of `None` in `model_parts`
* `intercept` parameter in `model_profile` is now named `center`

dalex 0.1.9
----------------------------------------------------------------
* *feature:* added `random_state` parameter for `predict_parts(type='shap')` and `model_profile` for reproducible calculations
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
* *feature:* `Exaplainer` attribute `model_info` remembers if parameters are default
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
