context("Check explain() function")

source("helper-objects.R")

test_that("Type of data in the explainer",{
  linear_model <- lm(m2.price ~ construction.year + surface + floor + no.rooms + district, data = apartments)

  explainer_lm0 <- explain(linear_model, colorize = FALSE)
  explainer_lm1 <- explain(linear_model)
  explainer_lm2 <- explain(linear_model, verbose = FALSE)
  explainer_lm4 <- explain(linear_model, data = apartments, label = "model_4v", y = apartments$m2.price)
  model_info <- list(package = "stats", ver = "3.6.1", type = "regression")
  explainer_lm5 <- explain(linear_model, data = apartments, label = "model_4v", y = apartments$m2.price, model_info = model_info)
  explainer_gbm <- explain(gbm1, data = apartments, y = apartments$m2.price)
  explainer_gbm2 <- explain(gbm2, data = titanic_imputed, y = titanic_imputed$survived)
  expect_true(is.data.frame(explainer_lm1$data))
  expect_is(explainer_lm1, "explainer")
  expect_is(explainer_lm2, "explainer")
  expect_is(explainer_lm4, "explainer")
  expect_is(explainer_gbm, "explainer")
  expect_is(explainer_gbm2, "explainer")
})

test_that("model_info work correctly", {
  mi1 <- model_info(model_classif_glm)
  mi2 <- model_info(model_classif_rf)
  mi3 <- model_info(model_regr_rf)
  mi4 <- model_info(model_regr_lm)
  mi5 <- model_info(gbm1)
  mi6 <- model_info(gbm2)

  expect_is(mi1, "model_info")
  expect_is(mi2, "model_info")
  expect_is(mi3, "model_info")
  expect_is(mi4, "model_info")
  expect_is(mi5, "model_info")
  expect_is(mi6, "model_info")
  expect_error(mi6, NA)
})

test_that("update_* work", {
  linear_model <- lm(m2.price ~ construction.year + surface + floor + no.rooms + district, data = apartments)
  explainer_lm0 <- explain(linear_model, colorize = FALSE)
  expect_is(update_label(explainer_lm0, "new_label"), "explainer")
  expect_is(update_data(explainer_lm0, apartments, as.data.frame(apartments$m2.price)), "explainer")
})
