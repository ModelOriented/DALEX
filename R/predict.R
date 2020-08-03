#' Predictions for the Explainer
#'
#' This is a generic \code{predict()} function works for \code{explainer} objects.
#'
#' @param object a model to be explained, object of the class \code{explainer}
#' @param explainer a model to be explained, object of the class \code{explainer}
#' @param newdata data.frame or matrix - observations for prediction
#' @param new_data data.frame or matrix - observations for prediction
#' @param ... other parameters that will be passed to the predict function
#'
#' @return An numeric matrix of predictions
#' @examples
#' HR_glm_model <- glm(status == "fired"~., data = HR, family = "binomial")
#' explainer_glm <- explain(HR_glm_model, data = HR)
#' predict(explainer_glm, HR[1:3,])
#'
#'  \donttest{
#' library("ranger")
#' HR_ranger_model <- ranger(status~., data = HR, num.trees = 50, probability = TRUE)
#' explainer_ranger  <- explain(HR_ranger_model, data = HR)
#' predict(explainer_ranger, HR[1:3,])
#'
#' model_prediction(explainer_ranger, HR[1:3,])
#'  }
#' @rdname predict
#' @export
predict.explainer <- function(object, newdata, ...) {
  model <- object$model
  predict_function <- object$predict_function
  predict_function(model, newdata, ...)
}

#' @rdname predict
#' @export
model_prediction  <- function(explainer, new_data, ...) {
  # this one will be deprecated
  model <- explainer$model
  predict_function <- explainer$predict_function
  predict_function(model, new_data, ...)
}

