context("Check model_performance() function")

mp_lm <- model_performance(explainer_regr_lm)
mp_rf <- model_performance(explainer_regr_rf)

test_that("Output format",{
  expect_is(mp_lm, "model_performance_explainer")
})

test_that("Output format - plot",{
  expect_is(plot(mp_lm, mp_rf), "gg")
  expect_is(plot(mp_lm, mp_rf, geom ="boxplot"), "gg")
})


test_that("include index option in boxplot showing outliers data index #34", {
  p_index <- plot(mp_lm, mp_rf, geom = "boxplot", show_outliers = 1, ptlabel = "index")
  p_name <- plot(mp_lm, mp_rf, geom = "boxplot", show_outliers = 1)
  nonindex_check <- rbind(mp_lm, mp_rf)
  nonindex_check$name <- rownames(nonindex_check)
  mp_lm$name <- seq.int(nrow(mp_lm))
  mp_rf$name <- seq.int(nrow(mp_rf))
  index_check <- rbind(mp_lm, mp_rf)

  expect_identical(index_check$name, p_index$data$name)
  expect_identical(nonindex_check$name, p_name$data$name)
  expect_error(plot(mp_lm, mp_rf, geom = "boxplot", show_outliers = 1, ptlabel = "asdf"), "The plot.model_performance() function requires label to be name or index.", fixed = TRUE)
})
