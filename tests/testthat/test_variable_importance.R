context("Check variable_importance() function")

vi_ranger <- variable_importance(explainer_regr_ranger, n_sample = 100)
vi_lm <- variable_importance(explainer_regr_lm, n_sample = 100)
vi_ranger_ratio <- variable_importance(explainer_regr_ranger, n_sample = 100, type = "ratio")


test_that("y not provided",{
  expect_error(variable_importance(explainer_regr_ranger_wo_y, n_sample = 100))
})

test_that("data not provided",{
  expect_error(variable_importance(explainer_wo_data, n_sample = 100))
})

test_that("wrong type value",{
  expect_error(variable_importance(explainer_regr_ranger, type="anything"))
})

test_that("Wrong object class (not explainer)", {
  expect_error(variable_importance(c(1,1)))
})

test_that("Output format - plot",{
  expect_is(plot(vi_ranger_ratio), "gg")
  expect_is(plot(vi_ranger, vi_lm), "gg")
  expect_is(plot(vi_lm), "gg")
})


test_that("Inverse sorting of bars",{
  expect_is(plot(vi_ranger_ratio, desc_sorting = FALSE), "gg")
  expect_is(plot(vi_ranger, vi_lm, desc_sorting = FALSE), "gg")
  expect_is(plot(vi_ranger, desc_sorting = FALSE), "gg")
})


