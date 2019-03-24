#' Wrapper over the predict function
#'
#' This function works for explain objects. It calles embeded predict function.
#'
#' @param object a model to be explained, object of the class 'explainer'
#' @param newdata data.frame or matrix - observations for prediction
#' @param ... other parameters that will be passed to the predict function
#'
#' @return An numeric matrix of predictions
#' @examples
#' HR_glm_model <- glm(status == "fired"~., data = HR, family = "binomial")
#' explainer_glm <- explain(HR_glm_model, data = HR)
#' predict(explainer_glm, HR[1:3,])
#'
#'  \dontrun{
#' library("randomForest")
#' HR_rf_model <- randomForest(status == "fired" ~., data = HR)
#' explainer_rf  <- explain(HR_rf_model, data = HR)
#' predict(explainer_rf, HR[1:3,])
#'  }
#' @rdname predict
#' @export
predict.explainer <- function(object, newdata, ...) {
  model <- object$model
  predict_function <- object$predict_function
  predict_function(model, newdata, ...)
}
