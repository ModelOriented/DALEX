context("Check variable_profile() function")


ranger_cp  <- variable_profile(explainer_classif_ranger, new_observation = titanic_imputed[1,-8], variables = "fare")
lm_cp <- variable_profile(explainer_regr_lm, new_observation = apartments[1,-1], variables = "surface")

test_that("Data wasn't provided", {
  expect_error(variable_profile(explainer_wo_data))

})

test_that("Wrong object class (not explainer)", {
  expect_error(variable_profile(c(1,1)))
})

test_that("Output format",{
  expect_is(ranger_cp, "ceteris_paribus_explainer")
  expect_is(lm_cp, "ceteris_paribus_explainer")
})

test_that("Output format - plot",{
  expect_is(plot(ranger_cp), "gg")
  expect_is(plot(lm_cp), "gg")
})
