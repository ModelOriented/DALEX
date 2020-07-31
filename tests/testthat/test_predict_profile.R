context("Check predict_profile() function")


ranger_cp  <- predict_profile(explainer_classif_ranger, new_observation = titanic_imputed[1,-8], variables = "fare")
lm_cp <- predict_profile(explainer_regr_lm, new_observation = apartments[1,-1], variables = "surface")

test_that("Data wasn't provided", {
  expect_error(predict_profile(explainer_wo_data))

})

test_that("Wrong object class (not explainer)", {
  expect_error(predict_profile(c(1,1)))
})

test_that("Output format",{
  expect_is(ranger_cp, c("ceteris_paribus_explainer", "predict_profile"))
  expect_is(lm_cp, c("ceteris_paribus_explainer", "predict_profile"))
})

test_that("Output format - plot",{
  expect_is(plot(ranger_cp), "gg")
  expect_is(plot(lm_cp), "gg")
})
