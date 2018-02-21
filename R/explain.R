#' Create Model Explainer
#'
#' Black-box models may have very different structures.
#' This function creates a unified representation of a model, which can be further processed by various explainers.
#'
#' @param model object - a model to be explained
#' @param data data.frame or marix - data that was used for fitting. If not provided then will be extracted from model fit
#' @param predict.function function that takes two arguments: model and new data and returns numeric vector with predictions
#' @param ... other parameters
#' @param label character - the name of the model. By default it's extracted from the 'class' attribute of the model
#'
#' @return An object of the class 'explainer'.
#'
#' It's a list with following fields:
#'
#' * \code{model} the explained model
#' * \code{data} the dataset used for training
#' * \code{predict.function} function that may be used for model predictions, shall return a single numerical value for each observation.
#' * \code{class} class/classess of a model
#' * \code{label} label, by default it's the last value from the \code{class} vector, but may be set to any character.
#'
#' @export
#' @importFrom stats predict
#' @importFrom utils head tail
#'
#' @examples
#' library("randomForest")
#' library("breakDown")
#'
#' wine_lm_model4 <- lm(quality ~ pH + residual.sugar + sulphates + alcohol, data = wine)
#' wine_lm_explainer4 <- explain(wine_lm_model4, data = wine, label = "model_4v")
#' wine_lm_explainer4
#'
#' wine_rf_model4 <- randomForest(quality ~ pH + residual.sugar + sulphates + alcohol, data = wine)
#' wine_rf_explainer4 <- explain(wine_rf_model4, data = wine, label = "model_rf")
#' wine_rf_explainer4
#'
explain <- function(model, data = NULL, y = NULL, predict.function = yhat, label = tail(class(model), 1)) {
  explainer <- list(model = model,
                    data = data,
                    y = y,
                    predict.function = predict.function,
                    class = class(model),
                    label = label)
  class(explainer) <- "explainer"
  explainer
}

yhat <- function(X.model, newdata) as.numeric(predict(X.model, newdata))
