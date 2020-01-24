context("Check variable_response() function")

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
  expect_true("data.frame" %in% class(vr_ale_rf))
})

test_that("Output format - plot",{
  expect_is(plot(vr_pdp_rf, vr_pdp_rf), "gg")
  expect_is(plot(vr_pdp_rf, vr_pdp_glm), "gg")
  expect_is(plot(vr_ale_rf, vr_ale_glm), "gg")
})
