context("Check prediction_breakdown() function")

new_apartments <- apartmentsTest[1001, -1]
pb_lm <- prediction_breakdown(explainer_regr_lm, observation = new_apartments)
pb_rf <- prediction_breakdown(explainer_regr_rf, observation = new_apartments)

test_that("Output format",{
  expect_is(pb_lm, "prediction_breakdown_explainer")
})

test_that("Output format - plot",{
  expect_is(plot(pb_rf, pb_lm), "gg")
})

