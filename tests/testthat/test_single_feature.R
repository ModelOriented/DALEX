context("Check feature_response_explainer() function")

library("randomForest")
library("DALEX")

HR_glm_model <- glm(status == "fired" ~ ., data = HR, family = "binomial")
explainer_glm <- explain(HR_glm_model, data = HR,  y = HR$status == "fired")

HR_rf_model <- randomForest(status ~ ., data = HR, ntree = 100)
explainer_rf  <- explain(HR_rf_model, data = HR,
                         y = as.data.frame(HR$status),
                         precalculate = FALSE, verbose = FALSE)

loss_cross_entropy <- function (observed, predicted, p_min = 0.0001) {
  p <- sapply(seq_along(observed), function(i) predicted[i, observed[i]])
  sum(-log(pmax(p, p_min)))
}


test_that("test age",{
  expl_glm <- feature_response(explainer_glm, "age", "pdp")
  expect_true("feature_response_explainer" %in% class(expl_glm))

  expl_rf <- feature_response(explainer_rf, "age", "pdp")
  expect_true("feature_response_explainer" %in% class(expl_rf))
})

test_that("test ale",{
  expl_glm <- feature_response(explainer_glm, "age", "ale")
  expect_true("feature_response_explainer" %in% class(expl_glm))
})
