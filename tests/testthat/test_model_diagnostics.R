context("Check model_diagnostics() function")

apartments_lm_model <- lm(m2.price ~ ., data = apartments)
explainer_lm <- explain(apartments_lm_model,
                         data = apartments,
                         y = apartments$m2.price, verbose = FALSE)
diag_lm <- model_diagnostics(explainer_lm)


library("ranger")
apartments_ranger_model <- ranger(m2.price ~ ., data = apartments)
explainer_ranger <- explain(apartments_ranger_model,
                         data = apartments,
                         y = apartments$m2.price, verbose = FALSE)
explainer_ranger_wo_precalculate <- explain(apartments_ranger_model,
                                            data = apartments,
                                            y = apartments$m2.price,
                                            precalculate = FALSE,
                                            verbose = FALSE)

diag_ranger_1 <- model_diagnostics(explainer_ranger)
diag_ranger_2 <- model_diagnostics(explainer_ranger, variables = c("surface", "construction.year"))
diag_ranger_3 <- model_diagnostics(explainer_ranger_wo_precalculate)



test_that("Choice of variables", {
  diag_ranger_1 <- model_diagnostics(explainer_ranger)
  expect_is(diag_ranger_1, "model_diagnostics")
  diag_ranger_2 <- model_diagnostics(explainer_ranger, variables = c("surface", "construction.year"))
  expect_is(diag_ranger_2, "model_diagnostics")
})

test_that("Explain without precalculations", {
  diag_ranger_3 <- model_diagnostics(explainer_ranger_wo_precalculate)
  expect_is(diag_ranger_3, "model_diagnostics")
})

test_that("Plot",{
  expect_is(plot(diag_ranger_1), "gg")
  expect_is(plot(diag_ranger_2), "gg")
  expect_is(plot(diag_ranger_3), "gg")
  expect_is(plot(diag_ranger_1, diag_lm, variable = "construction.year"), "gg")
})

test_that("Print",{
  expect_error(print(diag_ranger_1), NA)
  expect_error(print(diag_ranger_2), NA)
  expect_error(print(diag_ranger_3), NA)
})

test_that("Plot does not use deprecated aes_string() (#582)", {
  # ggplot2 >= 3.0.0 hard-deprecates aes_string(); plot.model_diagnostics()
  # used to call it directly, which raised a `lifecycle::deprecate_warn()`
  # every time a diagnostics plot was drawn (see reprex in issue #582).
  #
  # We check this structurally (source no longer references aes_string())
  # rather than by capturing the warning at call time, because lifecycle
  # warnings are only emitted once per R session: whichever test runs
  # plot.model_diagnostics() first "uses up" the warning for the rest of
  # the session/suite, which would make a warning-capturing test silently
  # pass regardless of whether the bug is actually fixed.
  plot_fun_src <- deparse(body(getS3method("plot", "model_diagnostics")))
  expect_false(any(grepl("aes_string", plot_fun_src, fixed = TRUE)))

  # the plot itself must still work and produce a ggplot object
  expect_is(plot(diag_ranger_1), "gg")
  expect_is(plot(diag_ranger_1, diag_lm, variable = "construction.year"), "gg")
})

explainer_array <- explainer_ranger
explainer_array$y_hat <- as.array(explainer_array$y_hat)

explainer_array2 <- explainer_ranger
explainer_array2$residuals <- as.array(explainer_array2$residuals)

test_that("array", {
  diag_array <- model_diagnostics(explainer_array)
  expect_is(diag_array, 'model_diagnostics')
  diag_array2 <- model_diagnostics(explainer_array2)
  expect_is(diag_array2, 'model_diagnostics')
})
