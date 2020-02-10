context("Check individual_diagnostics() function")


ranger_rd  <- individual_diagnostics(explainer_classif_ranger, new_observation = titanic_imputed[1,-8])
lm_rd <- individual_diagnostics(explainer_regr_lm, variables = c("surface", "floor"), new_observation = apartments[1,-1])

test_that("Data wasn't provided", {
  expect_error(individual_diagnostics(explainer_wo_data, new_observation = titanic_imputed[1,-8]))
})

test_that("new_observation has more than one row", {
  expect_error(individual_diagnostics(explainer_classif_ranger, new_observation = titanic_imputed[1:2,-8]))
})

test_that("Output format - plot",{
  expect_is(plot(ranger_rd), "gg")
  expect_is(plot(lm_rd), "gg")
})
