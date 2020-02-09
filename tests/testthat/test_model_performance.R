context("Check model_performance() function")

mp_lm <- model_performance(explainer_regr_lm)
mp_rf <- model_performance(explainer_regr_ranger)
mp_ran_classif <- model_performance(explainer_classif_ranger)
explainer_regr_lm_non_precalculate <- explain(model_regr_lm, data = apartments_test[1:1000, ],
                                              y = apartments_test$m2.price[1:1000], precalculate = FALSE, verbose = FALSE)

test_that("Output format",{
  expect_is(mp_lm, "model_performance_explainer")
  expect_is(mp_ran_classif, "model_performance_explainer")
  expect_error(print(mp_lm), NA)
})

test_that("If checks", {
  expect_error(model_performance("error"))
  data_null_explainer_regr_lm <- explainer_regr_lm
  data_null_explainer_regr_lm$data <- NULL
  expect_error(model_performance(data_null_explainer_regr_lm))
  y_null_explainer_regr_lm <- explainer_regr_lm
  y_null_explainer_regr_lm$y <- NULL
  y_null_explainer_regr_lm$y_hat <- NULL
  expect_error(model_performance(y_null_explainer_regr_lm))
  expect_error(model_performance(explainer_regr_lm_non_precalculate), NA)
})

test_that("Output format - plot",{
  expect_is(plot(mp_lm, mp_rf), "gg")
  expect_is(plot(mp_lm, mp_rf, mp_lm, mp_rf), "gg")
  expect_is(plot(mp_lm), "gg")
  expect_is(plot(mp_lm, mp_rf, geom ="boxplot"), "gg")
  expect_is(plot(mp_lm, mp_rf, geom ="gain"), "gg")
  expect_is(plot(mp_lm, mp_rf, geom ="lift"), "gg")
  expect_is(plot(mp_lm, mp_rf, geom ="histogram"), "gg")
  expect_is(plot(mp_ran_classif, mp_ran_classif, geom ="roc"), "gg")

})


test_that("include index option in boxplot showing outliers data index #34", {
  p_index <- plot(mp_lm, mp_rf, geom = "boxplot", show_outliers = 1, ptlabel = "index")
  p_name <- plot(mp_lm, mp_rf, geom = "boxplot", show_outliers = 1)
  nonindex_check <- rbind(mp_lm$residuals, mp_rf$residuals)
  nonindex_check$name <- rownames(nonindex_check)
  mp_lm$name <- seq.int(nrow(mp_lm$residuals))
  mp_rf$name <- seq.int(nrow(mp_rf$residuals))
  index_check <- rbind(mp_lm$residuals, mp_rf$residuals)

#  expect_identical(index_check$name, p_index$data$name)
  expect_identical(nonindex_check$name, p_name$data$name)
  expect_error(plot(mp_lm, mp_rf, geom = "boxplot", show_outliers = 1, ptlabel = "asdf"), "The plot.model_performance() function requires label to be name or index.", fixed = TRUE)
})
