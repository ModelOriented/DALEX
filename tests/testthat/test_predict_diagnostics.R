context("Check predict_diagnostics() function")


ranger_rd  <- predict_diagnostics(explainer_classif_ranger, new_observation = titanic_imputed[1,-8])
lm_rd <- predict_diagnostics(explainer_regr_lm, variables = c("surface", "floor"), new_observation = apartments[1,-1])


test_that("Data wasn't provided", {
  expect_error(predict_diagnostics(explainer_wo_data, new_observation = titanic_imputed[1,-8]))
})


explainer_regr_lm_data_cut <- explain(model_regr_lm, data = apartments_test[1:40, ], y = apartments_test$m2.price[1:40], verbose = FALSE)

test_that("Neighbors > explainer$data", {
  expect_warning(predict_diagnostics(explainer_regr_lm_data_cut, new_observation = apartments_test[41,]), "has to be lower than number of rows")
})

test_that("Output format - plot",{
  expect_is(plot(ranger_rd), "gg")
  expect_is(plot(lm_rd), "gg")
})


#:# alias

aranger_rd  <- individual_diagnostics(explainer_classif_ranger, new_observation = titanic_imputed[1,-8])
alm_rd <- individual_diagnostics(explainer_regr_lm, variables = c("surface", "floor"), new_observation = apartments[1,-1])

test_that("Data wasn't provided", {
  expect_error(individual_diagnostics(explainer_wo_data, new_observation = titanic_imputed[1,-8]))
})

test_that("Output format - plot",{
  expect_is(plot(aranger_rd), "gg")
  expect_is(plot(alm_rd), "gg")
})

test_that("Print",{
  expect_error(print(ranger_rd), NA)
  expect_error(print(lm_rd), NA)
})
