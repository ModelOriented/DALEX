dalex (development)
----------------------------------------------------------------
* *feature:* new `Explainer.residual` method which uses the
`residual_function` to calculate `residuals`
* *bugfix:* `Explainer.model_performance` method uses `self.model_type` when
 `model_type` is `None`

dalex 0.1.3
----------------------------------------------------------------
* [release](https://medium
.com/@ModelOriented/xai-in-python-with-dalex-4b173486aa92) of the `dalex` package
* `Explainer` object with `predict`, `predict_parts`, `predict_profile`,
`model_performance`, `model_parts` and `model_profile` methods
* `BreakDown`, `Shap`, `CeterisParibus`, `ModelPerformance`,
`VariableImportance` and `AggregatedProfiles` objects with `plot` method
* `load_titanic()` function which loads the `titanic_imputed` dataset
