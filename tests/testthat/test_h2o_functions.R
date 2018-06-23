library(h2o)
context("Support for h2o models")

h2o.init()
apartments_hf <- as.h2o(apartments)
model_h2o_glm <- h2o.glm(y = "m2.price", training_frame = apartments_hf)

explainer_h2o_glm <- explain(model = model_h2o_glm,
                             data = apartments_hf[,2:6],
                             y = apartments_hf$m2.price,
                             label = "h2o glm")

test_that("model_performance_h2o output",{
  expect_is(model_performance(explainer_h2o_glm), "model_performance_explainer")
})

test_that("variable_response_h2o output",{
  expect_is(variable_importance(explainer_h2o_glm, n_sample = 50), "variable_importance_explainer")
})

test_that("variable_response_h2o output",{
  expect_is(variable_response(explainer_h2o_glm, variable = "construction.year", type = "pdp", grid.resolution = 2),
"variable_response_explainer")
  expect_is(variable_response(explainer_h2o_glm, variable = "construction.year", type = "ale"),
            "variable_response_explainer")
  expect_is(variable_response(explainer_h2o_glm, variable = "district", type = "factor"),
            "variable_response_explainer")
})


new_apartment <- apartments_hf[1,]
test_that("variable_response_h2o output",{
  expect_is(prediction_breakdown(explainer_h2o_glm, observation = new_apartment),
            "prediction_breakdown_explainer")
})
