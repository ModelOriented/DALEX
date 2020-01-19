context("Check variable_attribution_break_down() function")

new_apartments <- apartments_test[1001, -1]
pb_lm_break_down <- variable_attribution(explainer_regr_lm, new_observation = new_apartments, type = "break_down")
pb_ranger_break_down <- variable_attribution(explainer_regr_ranger, new_observation = new_apartments, type = "break_down")
pb_lm_ibreak_down <- variable_attribution(explainer_regr_lm, new_observation = new_apartments, type = "ibreak_down")
pb_ranger_ibreak_down <- variable_attribution(explainer_regr_ranger, new_observation = new_apartments, type = "ibreak_down")
pb_lm_shap <- variable_attribution(explainer_regr_lm, new_observation = new_apartments, type = "shap")
pb_ranger_shap <- variable_attribution(explainer_regr_ranger, new_observation = new_apartments, type = "shap")


test_that("data not provided",{
  expect_error(variable_attribution(explainer_wo_data, type = "break_down"))
  expect_error(variable_attribution(explainer_wo_data, type = "ibreak_down"))
  expect_error(variable_attribution(explainer_wo_data, type = "shap"))
})

test_that("wrong type value",{
  expect_error(variable_attribution(explainer_regr_lm, new_observation = new_apartments, type = "b"))
})

test_that("Wrong object class (not explainer)", {
  expect_error(variable_attribution(list(1), type = "break_down"))
  expect_error(variable_attribution(list(1), type = "ibreak_down"))
  expect_error(variable_attribution(list(1), type = "shap"))
})

test_that("Output format",{
  expect_is(pb_lm_break_down, "break_down")
  expect_is(pb_ranger_break_down, "break_down")
  expect_is(pb_lm_ibreak_down, "break_down")
  expect_is(pb_ranger_ibreak_down, "break_down")
  expect_is(pb_lm_shap, "shap")
  expect_is(pb_ranger_shap, "shap")
})

test_that("Output format - plot",{
  expect_is(plot(pb_ranger_break_down), "gg")
  expect_is(plot(pb_ranger_break_down, pb_lm_break_down), "gg")
  expect_is(plot(pb_ranger_ibreak_down), "gg")
  expect_is(plot(pb_ranger_ibreak_down, pb_lm_ibreak_down), "gg")
  expect_is(plot(pb_ranger_shap), "gg")
  expect_is(plot(pb_ranger_shap, pb_lm_shap), "gg")
})

