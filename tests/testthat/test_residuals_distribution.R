context("Check residual_distribution() function")


ranger_rd  <- residuals_distribution(explainer_classif_ranger, new_observation = titanic_imputed[1,-8])
lm_rd <- residuals_distribution(explainer_regr_lm, variables = c("surface", "floor"), new_observation = apartments[1,-1])

test_that("Data wasn't provided", {
  expect_error(residuals_distribution(explainer_wo_data, new_observation = titanic_imputed[1,-8]))
})

test_that("new_observation has more than one row", {
  expect_error(residuals_distribution(explainer_classif_ranger, new_observation = titanic_imputed[1:2,-8]))
})

test_that("Output format - plot",{
  expect_is(ranger_rd, "gg")
  expect_is(lm_rd, "gg")
})
