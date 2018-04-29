
vi_rf <- variable_importance(explainer_regr_rf, n_sample = 100)
vi_lm <- variable_importance(explainer_regr_lm, n_sample = 100)
vi_rf_ratio <- variable_importance(explainer_regr_rf, n_sample = 100, type = "ratio")


test_that("y not provided",{
  expect_error(variable_importance(explainer_regr_rf_wo_y, n_sample = 100))
})

test_that("wrong type value",{
  expect_error(variable_importance(explainer_regr_rf, type="anything"))
})

test_that("variable_importance plots",{
  expect_is(plot(vi_rf_ratio), "gg")
  expect_is(plot(vi_rf, vi_lm), "gg")
})
