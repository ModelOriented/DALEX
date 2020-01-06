context("Check variable_attribution_break_down() function")

new_apartments <- apartments_test[1001, -1]
pb_lm <- variable_attribution_break_down(explainer_regr_lm, new_observation = new_apartments)
pb_rf <- variable_attribution_break_down(explainer_regr_rf, new_observation = new_apartments)

test_that("Output format",{
  expect_is(pb_lm, "break_down")
})

test_that("Output format - plot",{
  expect_is(plot(pb_rf, pb_lm), "gg")
})

