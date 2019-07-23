#' Create Model Explainer
#'
#' Black-box models may have very different structures.
#' This function creates a unified representation of a model, which can be further processed by various explainers.
#'
#' Please NOTE, that the \code{model} is the only required argument.
#' But some explainers may require that other arguments will be provided too.
#'
#' @param model object - a model to be explained
#' @param data data.frame or matrix - data that was used for fitting. If not provided then will be extracted from the model
#' @param y numeric vector with outputs / scores. If provided then it shall have the same size as \code{data}
#' @param predict_function function that takes two arguments: model and new data and returns numeric vector with predictions
#' @param residual_function function that takes three arguments: model, data and response vector y. It should return a numeric vector with model residuals for given data. If not provided, response residuals (\eqn{y-\hat{y}}) are calculated.
#' @param ... other parameters
#' @param label character - the name of the model. By default it's extracted from the 'class' attribute of the model
#' @param verbose if TRUE (default) then diagnostic messages will be printed
#'
#' @return An object of the class 'explainer'.
#'
#' It's a list with following fields:
#'
#' \itemize{
#' \item \code{model} the explained model
#' \item \code{data} the dataset used for training
#' \item \code{y} response for observations from \code{data}
#' \item \code{y_hat} calculated predictions
#' \item \code{residuals} calculated residuals
#' \item \code{predict_function} function that may be used for model predictions, shall return a single numerical value for each observation.
#' \item \code{residual_function} function that returns residuals, shall return a single numerical value for each observation.
#' \item \code{class} class/classes of a model
#' }
#'
#' @rdname explain
#' @export
#' @importFrom stats predict
#' @importFrom utils head tail installed.packages methods
#'
#' @examples
#' library("breakDown")
#'
#' wine_lm_model4 <- lm(quality ~ pH + residual.sugar + sulphates + alcohol, data = wine)
#' wine_lm_explainer4 <- explain(wine_lm_model4, data = wine, label = "model_4v")
#' wine_lm_explainer4
#'
#' wine_lm <- explain(wine_lm_model4)
#' wine_lm <- explain(wine_lm_model4, verbose = FALSE)
#' wine_lm <- explain(wine_lm_model4, data = wine, label = "model_4v", predict_function = predict)
#' wine_lm <- explain(wine_lm_model4, data = wine, label = "model_4v", y = wine$quality)
#' wine_lm <- explain(wine_lm_model4, data = wine, label = "model_4v", y = wine$quality,
#'                                    predict_function = predict)
#'
#'  \dontrun{
#' library("randomForest")
#' wine_rf_model4 <- randomForest(quality ~ pH + residual.sugar + sulphates + alcohol, data = wine)
#' wine_rf_explainer4 <- explain(wine_rf_model4, data = wine, label = "model_rf")
#' wine_rf_explainer4
#'  }
#'

#' @export
#' @rdname explain
explain.default <- function(model, data = NULL, y = NULL, predict_function = NULL, residual_function = NULL, ..., label = NULL, verbose = TRUE) {
  verbose_cat("Preparation of a new explainer is initiated\n", verbose = verbose)

  # REPORT: checks for model label
  if (is.null(label)) {
    # label not specified
    # try to extract something
    label <- tail(class(model), 1)
    verbose_cat("  -> model label       : ", label, " (default)\n", verbose = verbose)
  } else {
    verbose_cat("  -> model label       : ", label, "\n", verbose = verbose)
  }

  # REPORT: checks for data
  if (is.null(data)) {
    # data not specified
    # try to extract something
    possible_data <- try(model.frame(model), silent = TRUE)
    if (class(possible_data) != "try-error") {
      data <- possible_data
      verbose_cat("  -> data              : ", nrow(data), " rows ", ncol(data), " cols (extracted from model)\n", verbose = verbose)
    } else {
      verbose_cat("  -> no data avaliable! (WARNING)\n", verbose = verbose)
    }
  } else {
    verbose_cat("  -> data              : ", nrow(data), " rows ", ncol(data), " cols \n", verbose = verbose)
  }
  # as for issue #15, if data is in the tibble format then needs to be translated to data.frame
  if ("tbl" %in% class(data)) {
    data <- as.data.frame(data)
    verbose_cat("  -> data              :  tibbble converted into a data.frame \n", verbose = verbose)
  }

  # REPORT: checks for y
  if (is.null(y)) {
    # y not specified
    verbose_cat("  -> target variable   :  not specified! (WARNING)\n", verbose = verbose)
  } else {
    verbose_cat("  -> target variable   : ", length(y), " values \n", verbose = verbose)
    if (length(y) != nrow(data)) {
      verbose_cat("  -> target variable   :  length of 'y; is different than number of rows in 'data' (WARNING) \n", verbose = verbose)
    }
    if ((is.factor(y) | is.character(y))) {
      verbose_cat("  -> target variable   :  Please note that 'y' is a factor.", verbose = verbose)
      verbose_cat("  -> target variable   :  Consider changing the 'y' to a logical or numerical vector.", verbose = verbose)
    }
  }

  # REPORT: checks for predict_function
  if (is.null(predict_function)) {
    # predict_function not specified
    # try the default
    predict_function <- yhat
    # check if there is a specific version for this model
    yhat_functions <- methods("yhat")
    class(yhat_functions) = "character"
    matching_yhat <- intersect(paste0("yhat.", class(model)), yhat_functions)
    if (length(matching_yhat) == 0) {
      verbose_cat("  -> predict function  : yhat.default will be used (default)\n", verbose = verbose)
    } else {
      verbose_cat("  -> predict function  : ",matching_yhat[1]," will be used (default)\n", verbose = verbose)
    }
  } else {
    verbose_cat("  -> predict function  : ", deparse(substitute(predict_function)), "\n", verbose = verbose)
  }
  # if data is specified then we may test predict_function
  y_hat <- NULL
  if (!is.null(data) & verbose & !is.null(predict_function)) {
    y_hat <- predict_function(model, data)
    if ((is.factor(y_hat) | is.character(y_hat))) {
      verbose_cat("  -> predicted values  :  factor (WARNING) with levels: ", paste(unique(y_hat), collapse = ", "), "\n", verbose = verbose)
    } else {
      verbose_cat("  -> predicted values  :  numerical, min = ", min(y_hat), ", mean = ", mean(y_hat), ", max = ", max(y_hat), " \n", verbose = verbose)
    }
  }

  # REPORT: checks for residual_function
  if (is.null(residual_function)) {
    # residual_function not specified
    # try the default
    if (!is.null(predict_function)) {
      residual_function <- function(model, data, y) {
        y - predict_function(model, data)
      }
      verbose_cat("  -> residual function :  difference between y and yhat (default)\n", verbose = verbose)
    }
  } else {
    verbose_cat("  -> residual function : ", deparse(substitute(residual_function)), "\n", verbose = verbose)
  }
  # if data is specified then we may test residual_function
  residuals <- NULL
  if (!is.null(data) & !is.null(residual_function) & !is.null(y) & verbose) {
    residuals <- residual_function(model, data, y)
    if ((is.factor(residuals) | is.character(residuals))) {
      verbose_cat("  -> residuals         :  factor (WARNING) with levels: ", paste(unique(residuals), collapse = ", "), "\n", verbose = verbose)
    } else {
      verbose_cat("  -> residuals         :  numerical, min = ", min(residuals), ", mean = ", mean(residuals), ", max = ", max(residuals), " \n", verbose = verbose)
    }
  }

  # READY to create an explainer

  explainer <- list(model = model,
                    data = data,
                    y = y,
                    predict_function = predict_function,
                    y_hat = y_hat,
                    residuals = residuals,
                    class = class(model),
                    label = label)
  explainer <- c(explainer, list(...))
  class(explainer) <- "explainer"

  # everything went OK
  verbose_cat("A new explainer has been created!\n", verbose = verbose)
  explainer
}

verbose_cat <- function(..., verbose = TRUE) {
  if (verbose) {
    cat(...)
  }
}

#' @rdname explain
#' @export
explain <- explain.default

