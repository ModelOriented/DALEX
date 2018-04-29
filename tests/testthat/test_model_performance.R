mp_lm <- model_performance(explainer_regr_lm)
mp_rf <- model_performance(explainer_regr_rf)

test_that("model_performance plots",{
  expect_is(plot(mp_lm, mp_rf), "gg")
  expect_is(plot(mp_lm, mp_rf, geom ="boxplot"), "gg")
})
