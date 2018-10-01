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

test_that("Adding plot functionality to model_performance #34", {
  #original code
  model_performance <- function(explainer, ...) {
    if (!("explainer" %in% class(explainer))) stop("The model_performance() function requires an object created with explain() function.")
    if (is.null(explainer$data)) stop("The model_performance() function requires explainers created with specified 'data' parameter.")
    if (is.null(explainer$y)) stop("The model_performance() function requires explainers created with specified 'y' parameter.")

    observed <- explainer$y
    predicted <- explainer$predict_function(explainer$model, explainer$data, ...)
    residuals <- data.frame(predicted, observed, diff = predicted - observed)

    class(residuals) <- c("model_performance_explainer", "data.frame")
    residuals$label <- explainer$label
    residuals
  }

  old_mp_lm <- model_performance(explainer_regr_lm)
  old_mp_rf <- model_performance(explainer_regr_rf)
  expect_identical(colnames(mp_lm), c("predicted", "observed", "diff", "index", "label"))
  expect_identical(ncol(mp_lm), ncol(old_mp_lm)+1L)
  expect_identical(nrow(mp_lm), nrow(old_mp_lm))
  expect_identical(mp_lm$predicted, old_mp_lm$predicted)
  expect_identical(mp_lm$observed, old_mp_lm$observed)
  expect_identical(mp_lm$diff, old_mp_lm$diff)
  expect_identical(mp_lm$label, old_mp_lm$label)
  expect_identical(mp_lm$index, seq_along(old_mp_lm$label))
  expect_identical(colnames(mp_rf), c("predicted", "observed", "diff", "index", "label"))
  expect_identical(ncol(mp_rf), ncol(old_mp_rf)+1L)
  expect_identical(nrow(mp_rf), nrow(old_mp_rf))
  expect_identical(mp_rf$predicted, old_mp_rf$predicted)
  expect_identical(mp_rf$observed, old_mp_rf$observed)
  expect_identical(mp_rf$diff, old_mp_rf$diff)
  expect_identical(mp_rf$label, old_mp_rf$label)
  expect_identical(mp_rf$index, seq_along(old_mp_rf$label))
})

test_that("plot index versus mp idex issue #34", {
  p_lm <- plot(mp_lm, geom = "boxplot", show_outliers = 1)
  p_rf <- plot(mp_rf, geom = "boxplot", show_outliers = 1)
  expect_identical(mp_lm$index, p_lm$data$index)
  expect_identical(mp_rf$index, p_rf$data$index)
})
