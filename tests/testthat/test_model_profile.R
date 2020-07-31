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


#:# OLD FUNCTION NAMES

vr_pdp_rf  <- variable_effect(explainer_classif_ranger, variables = "age", type = "partial_dependency")
vr_pdp_glm  <- variable_effect(explainer_classif_glm, variables = "age", type = "partial_dependency")
vr_ale_rf  <- variable_effect(explainer_classif_ranger, variables = "age", type = "accumulated_dependency")
vr_ale_glm  <- variable_effect(explainer_classif_glm, variables = "age", type = "accumulated_dependency")

test_that("Data wasn't provided", {
  expect_error(variable_effect(explainer_wo_data, type = "partial_dependency"))
  expect_error(variable_effect(explainer_wo_data, type = "accumulated_dependency"))
})

test_that("Wrong object class (not explainer)", {
  expect_error(variable_effect(c(1,1), type = "partial_dependency"))
  expect_error(variable_effect(c(1,1), type = "accumulated_dependency"))
})

test_that("Unsupported type",{
  expect_error(variable_effect(explainer_classif_rf, variable = "age", type = "unknown"))
})

test_that("Non standard predict functions",{
  expect_true("data.frame" %in% class(vr_pdp_rf))
  expect_true("data.frame" %in% class(vr_ale_glm))
})

test_that("Output format - plot",{
  expect_is(plot(vr_pdp_rf, vr_ale_glm), "gg")
  expect_is(plot(vr_pdp_rf, vr_pdp_glm), "gg")
  expect_is(plot(vr_ale_rf, vr_ale_glm), "gg")
})
