#' Wrap Various Predict Functions
#'
#' This function is a wrapper over various predict functions for different models and differnt model structures.
#' The wrapper returns a single numeric score for each new observation.
#' To do this it uses different extraction techniques for models from different classes,
#' like for classification random forest is forces the output to be probabilities
#' not classes itself.
#'
#' Currently supported packages are:
#' \itemize{
#' \item class \code{cv.glmnet} and \code{glmnet} - models created with \pkg{glmnet} package,
#' \item class \code{glm} - generalized linear models created with \link[stats]{glm},
#' \item class \code{model_fit} - models created with \pkg{parsnip} package,
#' \item class \code{lm} - linear models created with \link[stats]{lm},
#' \item class \code{ranger} - models created with \pkg{ranger} package,
#' \item class \code{randomForest} - random forest models created with \pkg{randomForest} package,
#' \item class \code{svm} - support vector machines models created with the \pkg{e1071} package,
#' \item class \code{train} - models created with \pkg{caret} package,
#' \item class \code{gbm} - models created with \pkg{gbm} package,
#' \item class \code{lrm} - models created with \pkg{rms} package,
#' \item class \code{rpart} - models created with \pkg{rpart} package.
#' }
#'
#' @param X.model object - a model to be explained
#' @param newdata data.frame or matrix - observations for prediction
#' @param ... other parameters that will be passed to the predict function
#'
#' @return An numeric matrix of predictions
#'
#' @rdname yhat
#' @export
yhat <- function(X.model, newdata, ...)
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
    # if result is a vector then ncol parameter is null
    if (is.null(ncol(pred))) return(pred)
    #  binary classification
    if (!is.null(attr(X.model, "predict_function_target_column"))) return(pred[,attr(X.model, "predict_function_target_column")])
    if (ncol(pred) == 2 & is.null(attr(X.model, "predict_function_target_column"))) return(pred[,2])

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
    if (!is.null(attr(X.model, "predict_function_target_column"))) { # binary classification
      pred <- pred[,attr(X.model, "predict_function_target_column")]
    } else if (ncol(pred) == 2 & is.null(attr(X.model, "predict_function_target_column"))) {
      pred <- pred[,2]
    }
  } else {
    pred <- predict(X.model, newdata, ...)
  }
  pred
}

#' @rdname yhat
#' @export
yhat.gbm <- function(X.model, newdata, ...) {
  n.trees <- X.model$n.trees
  response <- predict(X.model, newdata = newdata, n.trees = n.trees, type = "response")
  #gbm returns and 3D array for multilabel classif
  if(length(dim(response)) > 2){
    if (!is.null(attr(X.model, "predict_function_target_column"))) {
      response <- response[,,1][,attr(X.model, "predict_function_target_column")]
    } else {
      response <- response[,,1]
    }
  }
  response
}


#' @rdname yhat
#' @export
yhat.glm <- function(X.model, newdata, ...) {
  predict(X.model, newdata, type = "response")
}

#' @rdname yhat
#' @export
yhat.cv.glmnet <- function(X.model, newdata, ...) {
  if (!"matrix" %in% class(newdata)) {
    newdata <- as.matrix(newdata)
  }
  if (!is.null(X.model$glmnet.fit$classnames)) {
    pred <- predict(X.model, newdata, type = "response", s = X.model$lambda[length(X.model$lambda)])
    #glmnet returns and 3D array for multilabel classif
    if(length(dim(pred)) > 2){
      return(pred[,,1])
    }
    if (ncol(pred) == 1) {
      return(as.numeric(pred))
    }
    if (!is.null(attr(X.model, "predict_function_target_column"))) {
      return(pred[,attr(X.model, "predict_function_target_column")])
    } else if (ncol(pred) == 2 & is.null(attr(X.model, "predict_function_target_column"))) {
      return(pred[,2])
    }
  } else {
    pred <- as.numeric(predict(X.model, newdata, type = "response", s = X.model$lambda[length(X.model$lambda)]))
  }
  pred
}

#' @rdname yhat
#' @export
yhat.glmnet <- function(X.model, newdata, ...) {
  if (!"matrix" %in% class(newdata)) {
    newdata <- as.matrix(newdata)
  }
  if (!is.null(X.model$classnames)) {
    pred <- predict(X.model, newdata, type = "response", s = X.model$lambda[length(X.model$lambda)])
    #glmnet returns and 3D array for multilabel classif
    if(length(dim(pred)) > 2){
      if (!is.null(attr(X.model, "predict_function_target_column"))) {
        return(pred[,,1][,attr(X.model, "predict_function_target_column")])
      } else {
        return(pred[,,1])
      }
    }
    # For binary classifiaction matrix with one column is returned
    if (ncol(pred) == 1) {
      return(as.numeric(pred))
    }
  } else {
    pred <- as.numeric(predict(X.model, newdata, type = "response", s = X.model$lambda[length(X.model$lambda)]))
  }
  pred
}

#' @rdname yhat
#' @export
yhat.ranger <- function(X.model, newdata, ...) {
  if (X.model$treetype == "Regression") {
    pred <- predict(X.model, newdata, ...)$predictions
  } else {
    # please note, that probability=TRUE should be set during training
    pred <- predict(X.model, newdata, ..., probability = TRUE)$predictions
    # if newdata has only one row then the vector needs to be transformed into a matrix
    if (nrow(newdata) == 1) {
      pred <- matrix(pred, nrow = 1)
      colnames(pred) <- colnames(X.model$predictions)
    }
    # if result is a vector then ncol parameter is null
    if (is.null(ncol(pred))) return(pred)
    # binary classification
    if (!is.null(attr(X.model, "predict_function_target_column"))) {
      pred <- pred[,attr(X.model, "predict_function_target_column")]
    } else if (ncol(pred) == 2 & is.null(attr(X.model, "predict_function_target_column"))) {
      pred <- pred[, 2]
    }
  }
  pred
}

#' @rdname yhat
#' @export
yhat.model_fit <- function(X.model, newdata, ...) {
  if (X.model$spec$mode == "classification") {
    response <- as.matrix(predict(X.model, newdata, type = "prob"))
    colnames(response) <- X.model$lvl
    if (!is.null(attr(X.model, "predict_function_target_column"))) {
      response <- response[,attr(X.model, "predict_function_target_column")]
    } else if (ncol(response) == 2  & is.null(attr(X.model, "predict_function_target_column"))) {
      response <- response[,2]
    }
  }
  if (X.model$spec$mode == "regression") {
    pred <- predict(X.model, newdata)
    response <- pred$.pred
  }
  response
}

#' @rdname yhat
#' @export
yhat.train <- function(X.model, newdata, ...) {
  if (X.model$modelType == "Classification") {
    response <- predict(X.model, newdata = newdata, type = "prob")
    if (!is.null(attr(X.model, "predict_function_target_column"))) {
      response <- response[,attr(X.model, "predict_function_target_column")]
    } else if (ncol(response) == 2 & is.null(attr(X.model, "predict_function_target_column"))) {
      response <- response[,2]
    }
  }
  if (X.model$modelType == "Regression") {
    response <- predict(X.model, newdata = newdata)

  }
  # fix for https://github.com/ModelOriented/DALEX/issues/150
  if (inherits(response, "data.frame")) response <- as.matrix(response)

  response
}


#' @rdname yhat
#' @export
yhat.lrm <- function(X.model, newdata, ...) {
  predict(X.model, newdata = newdata, type = "fitted")
}

#' @rdname yhat
#' @export
yhat.rpart <- function(X.model, newdata, ...) {
  response <- predict(X.model, newdata = newdata)
  if (!is.null(dim(response))) {
    if (!is.null(attr(X.model, "predict_function_target_column"))) {
      response <- response[,attr(X.model, "predict_function_target_column")]
    } else if (ncol(response) == 2 & is.null(attr(X.model, "predict_function_target_column"))) {
      response <- response[, 2]
    }
  }
  response
}

#' @rdname yhat
#' @export
yhat.function <- function(X.model, newdata, ...) {
  response <- X.model(newdata, ...)
  if (!is.null(dim(response))) {
    if (!is.null(attr(X.model, "predict_function_target_column"))) {
      response <- response[,attr(X.model, "predict_function_target_column")]
    } else if (ncol(response) == 2 & is.null(attr(X.model, "predict_function_target_column"))) {
      response <- response[, 2]
    }
  }
  response
}


#' @rdname yhat
#' @export
yhat.party <- function(X.model, newdata, ...) {
  yres <- class(X.model$fitted$`(response)`)
  if (yres[1] != "factor") {
    response <- predict(X.model, newdata, ...)
  } else {
    response <- predict(X.model, newdata, ..., type = "prob")
    if (!is.null(attr(X.model, "predict_function_target_column"))) {
      response <- response[,attr(X.model, "predict_function_target_column")]
    } else if (ncol(response) == 2 & is.null(attr(X.model, "predict_function_target_column"))) {
      response <- response[, 2]
    }
  }
  response
}

#' @rdname yhat
#' @export
yhat.default <- function(X.model, newdata, ...) {
  response <- predict(X.model, newdata, ...)
  # result is a vector
  if (is.null(dim(response))) {
    return(as.numeric(response))
  }
  # result is a matrix or data.frame with a single column
  if (ncol(response) == 1) {
    return(as.numeric(response))
  }
  # result is a matrix of data.frame with a two column (binary classification), returns the second
  if (!is.null(attr(X.model, "predict_function_target_column"))) {
    return(as.numeric(response[,attr(X.model, "predict_function_target_column")]))
  } else if (ncol(response) == 2 & is.null(attr(X.model, "predict_function_target_column"))) {
    return(as.numeric(response[,2]))
  }
  # result is a matrix of data.frame with more than 2 columns (multi label classification)
  as.matrix(response)
}




# #' @rdname yhat
# #' @export
# yhat.catboost.Model <- function(X.model, newdata, ...) {
#   newdata_pool <- catboost::catboost.load_pool(newdata)
#   catboost::catboost.predict(X.model, newdata_pool)
# }

