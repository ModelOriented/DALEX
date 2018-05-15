#' Create Model Explainer
#'
#' Black-box models may have very different structures.
#' This function creates a unified representation of a model, which can be further processed by various explainers.
#'
#' Please NOTE, that the \code{model} is actually the only required argument.
#' But some explainers may require that others will be provided too.
#'
#' @param model object - a model to be explained
#' @param data data.frame or matrix - data that was used for fitting. If not provided then will be extracted from the model
#' @param y numeric vector with outputs / scores. Currently used only by \code{variable_dropout()} explainer.
#' @param predict_function function that takes two arguments: model and new data and returns numeric vector with predictions
#' @param link function - a transformation/link function that shall be applied to raw model predictions
#' @param ... other parameters
#' @param label character - the name of the model. By default it's extracted from the 'class' attribute of the model
#'
#' @return An object of the class 'explainer'.
#'
#' It's a list with following fields:
#'
#' \itemize{
#' \item \code{model} the explained model
#' \item \code{data} the dataset used for training
#' \item \code{predict_function} function that may be used for model predictions, shall return a single numerical value for each observation.
#' \item \code{class} class/classes of a model
#' \item \code{label} label, by default it's the last value from the \code{class} vector, but may be set to any character.
#' }
#'
#' @rdname explain
#' @export
#' @importFrom stats predict
#' @importFrom utils head tail
#'
#' @examples
#' library("breakDown")
#'
#' wine_lm_model4 <- lm(quality ~ pH + residual.sugar + sulphates + alcohol, data = wine)
#' wine_lm_explainer4 <- explain(wine_lm_model4, data = wine, label = "model_4v")
#' wine_lm_explainer4
#'
#' # \dontrun{
#' library("randomForest")
#' wine_rf_model4 <- randomForest(quality ~ pH + residual.sugar + sulphates + alcohol, data = wine)
#' wine_rf_explainer4 <- explain(wine_rf_model4, data = wine, label = "model_rf")
#' wine_rf_explainer4
#' # }
#'
explain.default <- function(model, data = NULL, y = NULL, predict_function = yhat, link = I, ..., label = tail(class(model), 1)) {
  if (is.null(data)) {
    possible_data <- try(model.frame(model), silent = TRUE)
    if (class(possible_data) != "try-error") {
      data <- possible_data
    }
  }

  # as for issue #15, if data is in the tibble format then needs to be translated to data.frame
  if ("tbl" %in% class(data)) {
    data <- as.data.frame(data)
  }

  explainer <- list(model = model,
                    data = data,
                    y = y,
                    predict_function = predict_function,
                    link = link,
                    class = class(model),
                    label = label)
  explainer <- c(explainer, list(...))
  class(explainer) <- "explainer"
  explainer
}

#' @export
#' @rdname explain
explain <- explain.default

yhat <- function(X.model, newdata, ...) {
  if ("lm" %in% class(X.model)) {
    stats::predict(X.model, newdata, ...)
  } else {
    as.numeric(stats::predict(X.model, newdata, ...))
  }
}
