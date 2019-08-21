#' Create Model Explainer
#'
#' Black-box models may have very different structures.
#' This function creates a unified representation of a model, which can be further processed by various explainers.
#'
#' Please NOTE, that the \code{model} is the only required argument.
#' But some explainers may require that other arguments will be provided too.
#'
#' @param model object - a model to be explained
#' @param data data.frame or matrix - data that was used for fitting. If not provided then will be extracted from the model. Data should be passed without target column (y parameter). If not, some of the functionalities my not work.
#' @param y numeric vector with outputs / scores. If provided then it shall have the same size as \code{data}
#' @param predict_function function that takes two arguments: model and new data and returns numeric vector with predictions
#' @param residual_function function that takes three arguments: model, data and response vector y. It should return a numeric vector with model residuals for given data. If not provided, response residuals (\eqn{y-\hat{y}}) are calculated.
#' @param ... other parameters
#' @param label character - the name of the model. By default it's extracted from the 'class' attribute of the model
#' @param verbose if TRUE (default) then diagnostic messages will be printed
#' @param precalculate if TRUE (default) then 'predicted_values' and 'residuals' are calculated when explainer is created. This will happenn also if 'verbose' is TRUE
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
#'
#' aps_lm_model4 <- lm(m2.price ~., data = apartments)
#' aps_lm_explainer4 <- explain(aps_lm_model4, data = apartments, label = "model_4v")
#' aps_lm_explainer4
#'
#' aps_lm <- explain(aps_lm_model4)
#' aps_lm <- explain(aps_lm_model4, verbose = FALSE)
#' aps_lm <- explain(aps_lm_model4, data = apartments, label = "model_4v", predict_function = predict)
#' aps_lm <- explain(aps_lm_model4, data = apartments, label = "model_4v", y = apartments$m2.price)
#' aps_lm <- explain(aps_lm_model4, data = apartments, label = "model_4v", y = apartments$m2.price,
#'                                    predict_function = predict)
#'
#'  \dontrun{
#' library("randomForest")
#' aps_rf_model4 <- randomForest(m2.price ~., data = apartments)
#' aps_rf_explainer4 <- explain(aps_rf_model4, data = apartments, label = "model_rf")
#' aps_rf_explainer4
#'  }
#'

#' @export
#' @rdname explain
explain.default <- function(model, data = NULL, y = NULL, predict_function = NULL, residual_function = NULL, ..., label = NULL, verbose = TRUE, precalculate = TRUE) {
  verbose_cat("Preparation of a new explainer is initiated\n", verbose = verbose)

  # REPORT: checks for model label
  if (is.null(label)) {
    # label not specified
    # try to extract something
    label <- tail(class(model), 1)
    verbose_cat("  -> model label       : ", label, " (\033[33mdefault\033[39m)\n", verbose = verbose)
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
      verbose_cat("  -> no data avaliable! (\033[31mWARNING\033[39m)\n", verbose = verbose)
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
    verbose_cat("  -> target variable   :  not specified! (\033[31mWARNING\033[39m)\n", verbose = verbose)
  } else {
    if (is.data.frame(y)){
      y <- unlist(y, use.names = FALSE)
      verbose_cat("  -> target variable   :  Passed 'y' as data frame. Converted to a vector. (\033[90mNOTE\033[39m)\n", verbose = verbose)
    }
    verbose_cat("  -> target variable   : ", length(y), " values \n", verbose = verbose)
    if (length(y) != nrow(data)) {
      verbose_cat("  -> target variable   :  length of 'y; is different than number of rows in 'data' (\033[31mWARNING\033[39m) \n", verbose = verbose)
    }
    if ((is.factor(y) | is.character(y))) {
      verbose_cat("  -> target variable   :  Please note that 'y' is a factor.  (\033[31mWARNING\033[39m)\n", verbose = verbose)
      verbose_cat("  -> target variable   :  Consider changing the 'y' to a logical or numerical vector.\n", verbose = verbose)
      verbose_cat("  -> target variable   :  Otherwise I will not be able to calculate residuals or loss function.\n", verbose = verbose)
    }
    if (is_y_in_data(data, y)) {
      verbose_cat("  -> target variable   :  Column identical to `y` has been found in data.  (\033[31mWARNING\033[39m)\n", verbose = verbose)
      verbose_cat("  -> target variable   :  It is highly recommended to pass `data` without `y` column\n", verbose = verbose)
      verbose_cat("  -> target variable   :  Otherwise some functionalities may work in a inappropriate way \n", verbose = verbose)
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
      verbose_cat("  -> predict function  : yhat.default will be used (\033[33mdefault\033[39m)\n", verbose = verbose)
    } else {
      verbose_cat("  -> predict function  : ",matching_yhat[1]," will be used (\033[33mdefault\033[39m)\n", verbose = verbose)
    }
  } else {
    verbose_cat("  -> predict function  : ", deparse(substitute(predict_function)), "\n", verbose = verbose)
  }
  # if data is specified then we may test predict_function
  y_hat <- NULL
  if (!is.null(data) & !is.null(predict_function) & (verbose | precalculate)) {
    y_hat <- try(predict_function(model, data), silent = TRUE)
    if (class(y_hat) == "try-error") {
      y_hat <- NULL
      verbose_cat("  -> predicted values  :  the predict_function returns an error when executed (\033[31mWARNING\033[39m) \n", verbose = verbose)
    } else {
      if ((is.factor(y_hat) | is.character(y_hat))) {
        verbose_cat("  -> predicted values  :  factor (\033[31mWARNING\033[39m) with levels: ", paste(unique(y_hat), collapse = ", "), "\n", verbose = verbose)
      } else {
        verbose_cat("  -> predicted values  :  numerical, min = ", min(y_hat), ", mean = ", mean(y_hat), ", max = ", max(y_hat), " \n", verbose = verbose)
      }
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
      verbose_cat("  -> residual function :  difference between y and yhat (\033[33mdefault\033[39m)\n", verbose = verbose)
    }
  } else {
    verbose_cat("  -> residual function : ", deparse(substitute(residual_function)), "\n", verbose = verbose)
  }
  # if data is specified then we may test residual_function
  residuals <- NULL
  if (!is.null(data) & !is.null(residual_function) & !is.null(y) & (verbose | precalculate)) {
    residuals <- try(residual_function(model, data, y), silent = TRUE)
    if (class(residuals) == "try-error") {
      residuals <- NULL
      verbose_cat("  -> residuals         :  the residual_function returns an error when executed (\033[31mWARNING\033[39m) \n", verbose = verbose)
    } else {
      if ((is.factor(residuals) | is.character(residuals))) {
        verbose_cat("  -> residuals         :  factor (\033[31mWARNING\033[39m) with levels: ", paste(unique(residuals), collapse = ", "), "\n", verbose = verbose)
      } else {
        verbose_cat("  -> residuals         :  numerical, min = ", min(residuals), ", mean = ", mean(residuals), ", max = ", max(residuals), " \n", verbose = verbose)
      }
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
  verbose_cat("\033[32mA new explainer has been created!\033[39m\n", verbose = verbose)
  explainer
}

verbose_cat <- function(..., verbose = TRUE) {
  if (verbose) {
    cat(...)
  }
}

is_y_in_data <- function(data, y) {
  any(sapply(data, function(x) {
    all(as.character(x) == as.character(y))
  }))
}

#' @rdname explain
#' @export
explain <- explain.default

