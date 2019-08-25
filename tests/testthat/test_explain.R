context("Check explain() function")

test_that("Type of data in the explainer",{
  linear_model <- lm(m2.price ~ construction.year + surface + floor + no.rooms + district, data = apartments)

  explainer_lm1 <- explain(linear_model)
  explainer_lm2 <- explain(linear_model, verbose = FALSE)
  explainer_lm3 <- explain(linear_model, data = apartments, label = "model_4v", predict_function = DALEX::yhat.randomForest)
  explainer_lm4 <- explain(linear_model, data = apartments, label = "model_4v", y = apartments$m2.price)
  explainer_lm5 <- explain(linear_model, data = apartments, label = "model_4v", y = apartments$m2.price,
                           predict_function = DALEX::yhat.randomForest)

  expect_true(is.data.frame(explainer_lm1$data))
  expect_is(explainer_lm1, "explainer")
  expect_is(explainer_lm2, "explainer")
  expect_is(explainer_lm3, "explainer")
  expect_is(explainer_lm4, "explainer")
  expect_is(explainer_lm5, "explainer")
})
