test_that("Non standard predict functions",{
  library(breakDown)
  library(randomForest)
  HR_rf_model <- randomForest(factor(left)~., data = breakDown::HR_data, ntree = 100)
  explainer_rf  <- explain(HR_rf_model, data = HR_data,
                           predict_function = function(model, x) predict(model, x, type = "prob")[,2])
  expl_rf  <- variable_response(explainer_rf, variable = "satisfaction_level", type = "pdp")
  expect_true("data.frame" %in% class(expl_rf))

  expl_rf  <- variable_response(explainer_rf, variable = "satisfaction_level", type = "ale")
  expect_true("data.frame" %in% class(expl_rf))
})
