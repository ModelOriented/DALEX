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
#'  \dontrun{
#' library("randomForest")
#' wine_rf_model4 <- randomForest(quality ~ pH + residual.sugar + sulphates + alcohol, data = wine)
#' wine_rf_explainer4 <- explain(wine_rf_model4, data = wine, label = "model_rf")
#' wine_rf_explainer4
#'  }
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

#' Wrapper over the predict function
#'
#' This function is just a wrapper over the predict function.
#' It sets different default parameters for models from different classes,
#' like for classification random Forest is forces the output to be probabilities not classes itself.
#'
#' @param X.model object - a model to be explained
#' @param newdata data.frame or matrix - observations for prediction
#' @param ... other parameters that will be passed to the predict function
#'
#' @return An numeric matrix of predictions
#'
#' @rdname yhat
#' @export
yhat <- function (X.model, newdata, ...)
  UseMethod("yhat")

#' @rdname yhat
#' @export
yhat.lm <- function(X.model, newdata, ...) {
  predict(X.model, newdata, ...)
}

#' @rdname yhat
#' @export
yhat.randomForest <- function(X.model, newdata, ...) {
  if (X.model$type == "classification") {
    pred <- predict(X.model, newdata, type = "prob", ...)
    if (ncol(pred) == 2) { # binary classification
      pred <- pred[,2]
    }
  } else {
    pred <- predict(X.model, newdata, ...)
  }
  pred
}

#' @rdname yhat
#' @export
yhat.svm <- function(X.model, newdata, ...) {
  if (X.model$type == 0) {
    pred <- attr(predict(X.model, newdata = newdata, probability = TRUE), "probabilities")
    if (ncol(pred) == 2) { # binary classification
      pred <- pred[,2]
    }
  } else {
    pred <- predict(X.model, newdata, ...)
  }
  pred
}

#' @rdname yhat
#' @export
yhat.glm <- function(X.model, newdata, ...) {
  predict(X.model, newdata, type = "response")
}


#' @rdname yhat
#' @export
yhat.ranger <- function(X.model, newdata, ...) {
  if (X.model$treetype == "Regression") {
    pred <- predict(X.model, newdata, ...)$predictions
  } else {
    # please note, that probability=TRUE should be set during training
    pred <- predict(X.model, newdata, ..., probability = TRUE)$predictions
    if (ncol(pred) == 2) { # binary classification
      pred <- pred[,2]
    }
  }
  pred
}

#' @rdname yhat
#' @export
yhat.default <- function(X.model, newdata, ...) {
  as.numeric(predict(X.model, newdata, ...))
}
