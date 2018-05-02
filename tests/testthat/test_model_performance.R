context("Check model_performance() function")

mp_lm <- model_performance(explainer_regr_lm)
mp_rf <- model_performance(explainer_regr_rf)

test_that("Output format",{
  expect_is(mp_lm, "model_performance_explainer")
})

test_that("Output format - plot",{
  expect_is(plot(mp_lm, mp_rf), "gg")
  expect_is(plot(mp_lm, mp_rf, geom ="boxplot"), "gg")
  expect_is(plot(mp_lm, mp_rf, geom = "point"), "gg")
})
