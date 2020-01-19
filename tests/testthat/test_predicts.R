context("Check predict.explainer() function")

test_that("Output format",{
  expect_is(model_prediction(explainer_classif_ranger, titanic_imputed), "numeric")
  expect_is(predict(explainer_classif_ranger, titanic_imputed), "numeric")
})
