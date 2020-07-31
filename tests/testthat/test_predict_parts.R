context("Check predict_parts() function")

new_apartments <- apartments_test[1001, -1]
pp_lm_break_down <- predict_parts(explainer_regr_lm, new_observation = new_apartments, type = "break_down")
pp_ranger_break_down <- predict_parts(explainer_regr_ranger, new_observation = new_apartments, type = "break_down")
pp_lm_ibreak_down <- predict_parts(explainer_regr_lm, new_observation = new_apartments, type = "break_down_interactions")
pp_ranger_ibreak_down <- predict_parts(explainer_regr_ranger, new_observation = new_apartments, type = "break_down_interactions")
pp_lm_shap <- predict_parts(explainer_regr_lm, new_observation = new_apartments, type = "shap")
pp_ranger_shap <- predict_parts(explainer_regr_ranger, new_observation = new_apartments, type = "shap")

pp_lm_osc <- predict_parts(explainer_regr_lm, new_observation = new_apartments, type = "oscillations")
pp_ranger_osc <- predict_parts(explainer_regr_ranger, new_observation = new_apartments, type = "oscillations")
pp_lm_osc_uni <- predict_parts(explainer_regr_lm, new_observation = new_apartments, type = "oscillations_uni")
pp_ranger_osc_uni <- predict_parts(explainer_regr_ranger, new_observation = new_apartments, type = "oscillations_uni")
pp_lm_osc_emp <- predict_parts(explainer_regr_lm, new_observation = new_apartments, type = "oscillations_emp")
pp_ranger_osc_emp <- predict_parts(explainer_regr_ranger, new_observation = new_apartments, type = "oscillations_emp")


test_that("data not provided",{
  expect_error(predict_parts(explainer_wo_data, type = "break_down"))
  expect_error(predict_parts(explainer_wo_data, type = "break_down_interactions"))
  expect_error(predict_parts(explainer_wo_data, type = "shap"))
})

test_that("wrong type value",{
  expect_error(predict_parts(explainer_regr_lm, new_observation = new_apartments, type = "b"))
})

test_that("Wrong object class (not explainer)", {
  expect_error(predict_parts(list(1), type = "break_down"))
  expect_error(predict_parts(list(1), type = "break_down_interactions"))
  expect_error(predict_parts(list(1), type = "shap"))
})

test_that("Output format",{
  expect_is(pp_lm_break_down, c("break_down", 'predict_parts'))
  expect_is(pp_ranger_break_down, c("break_down", 'predict_parts'))
  expect_is(pp_lm_ibreak_down, c("break_down", 'predict_parts'))
  expect_is(pp_ranger_ibreak_down, c("break_down", 'predict_parts'))
  expect_is(pp_lm_shap, c("shap", 'predict_parts'))
  expect_is(pp_ranger_shap, c("shap", 'predict_parts'))
  expect_is(pp_lm_osc, c("oscillations", 'predict_parts'))
  expect_is(pp_ranger_osc, c("oscillations", 'predict_parts'))
  expect_is(pp_lm_osc_uni, c("oscillations_uni", 'predict_parts'))
  expect_is(pp_ranger_osc_uni, c("oscillations_uni", 'predict_parts'))
  expect_is(pp_lm_osc_emp, c("oscillations_emp", 'predict_parts'))
  expect_is(pp_ranger_osc_emp, c("oscillations_emp", 'predict_parts'))
})

test_that("Output format - plot",{
  expect_is(plot(pp_ranger_break_down), "gg")
  expect_is(plot(pp_ranger_break_down, pp_lm_break_down), "gg")
  expect_is(plot(pp_ranger_ibreak_down), "gg")
  expect_is(plot(pp_ranger_ibreak_down, pp_lm_ibreak_down), "gg")
  expect_is(plot(pp_ranger_shap), "gg")
  expect_is(plot(pp_ranger_shap, pp_lm_shap), "gg")
  expect_is(plot(pp_lm_osc), "gg")
  expect_is(plot(pp_ranger_osc), "gg")
  expect_is(plot(pp_lm_osc_uni), "gg")
  expect_is(plot(pp_ranger_osc_uni), "gg")
  expect_is(plot(pp_lm_osc_emp), "gg")
  expect_is(plot(pp_ranger_osc_emp), "gg")
})

