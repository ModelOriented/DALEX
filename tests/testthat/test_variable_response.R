context("Check variable_response() function")

vr_pdp_rf  <- variable_response(explainer_classif_rf, variable = "age", type = "pdp",
                                trans = function(x) exp(x))
vr_pdp_glm  <- variable_response(explainer_classif_glm, variable = "age", type = "pdp")
vr_ale_rf  <- variable_response(explainer_classif_rf, variable = "age", type = "ale")
vr_ale_glm  <- variable_response(explainer_classif_glm, variable = "age", type = "ale")


test_that("Data wasn't provided", {
  expect_error(variable_response(explainer_wo_data))
})

test_that("Unsupported type",{
  expect_error(variable_response(explainer_classif_rf, variable = "age", type = "unknown"))
})

test_that("Non standard predict functions",{
  expect_true("data.frame" %in% class(vr_pdp_rf))
  expect_true("data.frame" %in% class(vr_ale_rf))
})

test_that("Output format - plot",{
  expect_is(plot(vr_pdp_rf, vr_pdp_rf), "gg")
  expect_is(plot(vr_pdp_rf, vr_pdp_glm), "gg")
  expect_is(plot(vr_ale_rf, vr_ale_glm), "gg")
  expect_is(plot(vr_factor_rf, vr_factor_lm), "gg")
})
