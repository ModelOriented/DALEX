#' Explanations for a Single Prediction
#'
#' @param explainer a model to be explained, preprocessed by the 'explain' function
#' @param observation a new observarvation for which predictions need to be explained
#' @param ... other parameters
#'
#' @return An object of the class 'single_prediction_explainer'.
#' It's a data frame with calculated average response.
#'
#' @export
#' @import breakDown
#' @examples
#' library("randomForest")
#' library("breakDown")
#' new.wine <- data.frame(citric.acid = 0.35,
#'      sulphates = 0.6,
#'      alcohol = 12.5,
#'      pH = 3.36,
#'      residual.sugar = 4.8)
#'
#' wine_lm_model4 <- lm(quality ~ pH + residual.sugar + sulphates + alcohol, data = wine)
#' wine_lm_explainer4 <- explain(wine_lm_model4, data = wine, label = "model_4v")
#' wine_lm_predict4 <- single_prediction(wine_lm_explainer4, observation = new.wine)
#' wine_lm_predict4
#'
#' wine_rf_model4 <- randomForest(quality ~ pH + residual.sugar + sulphates + alcohol, data = wine)
#' wine_rf_explainer4 <- explain(wine_rf_model4, data = wine, label = "model_rf")
#' wine_rf_predict4 <- single_prediction(wine_rf_explainer4, observation = new.wine)
#' wine_rf_predict4
#'
single_prediction <- function(explainer, observation, ...) {
  stopifnot(class(explainer) == "explainer")

  # breakDown
  res <- broken(explainer$model,
                new_observation = observation,
                data = explainer$data,
                baseline = "Intercept", ...)
  res$label <- rep(explainer$label, length(res$variable))

  class(res) <- c("single_prediction_explainer", "data.frame")
  res

}


