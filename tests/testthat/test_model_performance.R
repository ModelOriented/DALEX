context("Check model_performance() function")

mp_lm <- model_performance(explainer_regr_lm)
mp_rf <- model_performance(explainer_regr_rf)

test_that("y not provided",{
  expect_error(model_performance(explainer_regr_rf_wo_y))
})
test_that("data not provided",{
  expect_error(model_performance(explainer_regr_wo_data))
})
test_that("explainer class",{
  expect_error(model_performance(c(1,2,3)))
})


test_that("Output format",{
  expect_is(mp_lm, "model_performance_explainer")
})

test_that("Output format - plot",{
  expect_is(plot(mp_lm, mp_rf), "gg")
  expect_is(plot(mp_lm, mp_rf, geom ="boxplot"), "gg")
})
