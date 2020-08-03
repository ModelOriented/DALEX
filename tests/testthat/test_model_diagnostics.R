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

