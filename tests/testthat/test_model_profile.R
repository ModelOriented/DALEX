context("Check model_profile() function")

mp_pdp_rf  <- model_profile(explainer_classif_ranger, variables = "age", type = "partial")
mp_pdp_glm  <- model_profile(explainer_classif_glm, variables = "age", type = "partial")
mp_ale_rf  <- model_profile(explainer_classif_ranger, variables = "age", type = "accumulated")
mp_ale_glm  <- model_profile(explainer_classif_glm, variables = "age", type = "accumulated")


test_that("Data wasn't provided", {
  expect_error(model_profile(explainer_wo_data, type = "partial"))
  expect_error(model_profile(explainer_wo_data, type = "accumulated"))
})

test_that("Wrong object class (not explainer)", {
  expect_error(model_profile(c(1,1), type = "partial"))
  expect_error(model_profile(c(1,1), type = "accumulated"))
})

test_that("Unsupported type",{
  expect_error(model_profile(explainer_classif_rf, variable = "age", type = "unknown"))
})

test_that("Non standard predict functions",{
  expect_is(mp_pdp_rf, 'model_profile')
  expect_is(mp_ale_glm, 'model_profile')
})

test_that("Output format - plot",{
  expect_is(plot(mp_pdp_rf$agr_profiles, mp_pdp_glm$agr_profiles), "gg")
  expect_is(plot(mp_pdp_rf$agr_profiles, mp_pdp_glm$agr_profiles), "gg")
  expect_is(plot(mp_ale_rf$agr_profiles, mp_ale_glm$agr_profiles), "gg")
})
