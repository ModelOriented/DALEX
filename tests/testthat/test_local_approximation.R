context("Check prediction_breakdown() function")

new_apartments <- apartmentsTest[1001, ]
pa_rf <- prediction_approximation(explainer_regr_rf, observation = new_apartments,
                                  "m2.price", 500)
test_that("Output format",{
  expect_is(pa_rf, "prediction_approximation_explainer")
})

test_that("Generic functions",{
  expect_is(plot(pa_rf, type = "forest"), "gg")
  expect_is(plot(pa_rf, type = "waterfall"), "gg")
  expect_output(print(pa_rf))
})

