#' Create Model Explainer
#'
#' Black-box models may have very different structures.
#' This function creates a unified representation of a model, which can be further processed by various explainers.
#'
#' Please NOTE, that the \code{model} is the only required argument.
#' But some explainers may require that other arguments will be provided too.
#'
#' @param model object - a model to be explained
#' @param data data.frame or matrix - data that was used for fitting. If not provided then will be extracted from the model. Data should be passed without target column (this shall be provided as the \code{y} argument). NOTE: If target variable is present in the \code{data}, some of the functionalities my not work properly.
#' @param y numeric vector with outputs / scores. If provided then it shall have the same size as \code{data}
#' @param weights numeric vector with sampling weights. By default it's \code{NULL}. If provided then it shall have the same length as \code{data}
#' @param predict_function function that takes two arguments: model and new data and returns numeric vector with predictions
#' @param residual_function function that takes three arguments: model, data and response vector y. It should return a numeric vector with model residuals for given data. If not provided, response residuals (\eqn{y-\hat{y}}) are calculated.
#' @param ... other parameters
#' @param label character - the name of the model. By default it's extracted from the 'class' attribute of the model
#' @param verbose if TRUE (default) then diagnostic messages will be printed
#' @param precalculate if TRUE (default) then \code{predicted_values} and \code{residual} are calculated when explainer is created.
#' This will happen also if \code{verbose} is TRUE. Set both \code{verbose} and \code{precalculate} to FALSE to omit calculations.
#' @param colorize if TRUE (default) then \code{WARNINGS}, \code{ERRORS} and \code{NOTES} are colorized. Will work only in the R console.
#' @param model_info a named list (\code{package}, \code{version}, \code{type}) containg information about model. If \code{NULL}, \code{DALEX} will seek for information on it's own.
#'
#' @return An object of the class \code{explainer}.
#'
#' It's a list with following fields:
#'
#' \itemize{
#' \item \code{model} the explained model.
#' \item \code{data} the dataset used for training.
#' \item \code{y} response for observations from \code{data}.
#' \item \code{weights} sample weights for \code{data}. \code{NULL} if weights are not specified.
#' \item \code{y_hat} calculated predictions.
#' \item \code{residuals} calculated residuals.
#' \item \code{predict_function} function that may be used for model predictions, shall return a single numerical value for each observation.
#' \item \code{residual_function} function that returns residuals, shall return a single numerical value for each observation.
#' \item \code{class} class/classes of a model.
#' \item \code{label} label of explainer.
#' \item \code{model_info} named list contating basic information about model, like package, version of package and type.
#' }
#'
#' @rdname explain
#' @export
#' @importFrom stats predict
#' @importFrom utils head tail installed.packages methods
#'
#' @examples
#' # simple explainer for regression problem
#' aps_lm_model4 <- lm(m2.price ~., data = apartments)
#' aps_lm_explainer4 <- explain(aps_lm_model4, data = apartments, label = "model_4v")
#' aps_lm_explainer4
#'
#' # various parameters for the explain function
#' # all defaults
#' aps_lm <- explain(aps_lm_model4)
#'
#' # silent execution
#' aps_lm <- explain(aps_lm_model4, verbose = FALSE)
#'
#' # user provided predict_function
#' aps_lm <- explain(aps_lm_model4, data = apartments, label = "model_4v", predict_function = predict)
#'
#' # set target variable
#' aps_lm <- explain(aps_lm_model4, data = apartments, label = "model_4v", y = apartments$m2.price)
#' aps_lm <- explain(aps_lm_model4, data = apartments, label = "model_4v", y = apartments$m2.price,
#'                                    predict_function = predict)
#' # set model_info
#' model_info <- list(package = "stats", ver = "3.6.1", type = "regression")
#' aps_lm_model4 <- lm(m2.price ~., data = apartments)
#' aps_lm_explainer4 <- explain(aps_lm_model4, data = apartments, label = "model_4v",
#'                              model_info = model_info)
#'
#' \dontrun{
#' # set model_info
#' model_info <- list(package = "stats", ver = "3.6.1", type = "regression")
#' aps_lm_model4 <- lm(m2.price ~., data = apartments)
#' aps_lm_explainer4 <- explain(aps_lm_model4, data = apartments, label = "model_4v",
#'                              model_info = model_info)
#'
#' aps_lm_explainer4 <- explain(aps_lm_model4, data = apartments, label = "model_4v",
#'                              weights = as.numeric(apartments$construction.year > 2000))
#'
#' # more complex model
#' library("randomForest")
#' aps_rf_model4 <- randomForest(m2.price ~., data = apartments)
#' aps_rf_explainer4 <- explain(aps_rf_model4, data = apartments, label = "model_rf")
#' aps_rf_explainer4
#'  }
#'
#' @export
explain.default <- function(model, data = NULL, y = NULL, predict_function = NULL,
                            residual_function = NULL, weights = NULL, ...,
                            label = NULL, verbose = TRUE, precalculate = TRUE,
                            colorize = TRUE, model_info = NULL) {

  verbose_cat("Preparation of a new explainer is initiated\n", verbose = verbose)

  # if requested, remove colors
  if (!colorize) {
    color_codes <- list(yellow_start = "", yellow_end = "",
                        red_start = "", red_end = "",
                        green_start = "", green_end = "")
  }

  # REPORT: checks for model label
  if (is.null(label)) {
    # label not specified
    # try to extract something
    label <- tail(class(model), 1)
    verbose_cat("  -> model label       : ", label, " (",color_codes$yellow_start,"default",color_codes$yellow_end,")\n", verbose = verbose)
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
      verbose_cat("  -> data              : ", nrow(data), " rows ", ncol(data), " cols (\033[33mextracted from the model\033[39m)\n", verbose = verbose)
    } else {
      verbose_cat("  -> no data avaliable! (",color_codes$red_start,"WARNING",color_codes$red_end,")\n", verbose = verbose)
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
    verbose_cat("  -> target variable   :  not specified! (",color_codes$red_start,"WARNING",color_codes$red_end,")\n", verbose = verbose)
  } else {
    if (is.data.frame(y)) {
      y <- unlist(y, use.names = FALSE)
      verbose_cat("  -> target variable   :  Argument 'y' was a data frame. Converted to a vector. (",color_codes$red_start,"WARNING",color_codes$red_end,")\n", verbose = verbose)
    }
    verbose_cat("  -> target variable   : ", length(y), " values \n", verbose = verbose)
    if (length(y) != nrow(data)) {
      verbose_cat("  -> target variable   :  length of 'y' is different than number of rows in 'data' (",color_codes$red_start,"WARNING",color_codes$red_end,") \n", verbose = verbose)
    }
    if ((is.factor(y) | is.character(y))) {
      verbose_cat("  -> target variable   :  Please note that 'y' is a factor.  (",color_codes$red_start,"WARNING",color_codes$red_end,")\n", verbose = verbose)
      verbose_cat("  -> target variable   :  Consider changing the 'y' to a logical or numerical vector.\n", verbose = verbose)
      verbose_cat("  -> target variable   :  Otherwise I will not be able to calculate residuals or loss function.\n", verbose = verbose)
    }

    if (!is.null(data) & is_y_in_data(data, y)) {
      verbose_cat("  -> data              :  A column identical to the target variable `y` has been found in the `data`.  (",color_codes$red_start,"WARNING",color_codes$red_end,")\n", verbose = verbose)
      verbose_cat("  -> data              :  It is highly recommended to pass `data` without the target variable column\n", verbose = verbose)
    }
  }

  # REPORT: checks for weights
  if (is.null(weights)) {
    # weights not specified
    # do nothing
  } else {
    if (is.data.frame(weights)) {
      weights <- unlist(weights, use.names = FALSE)
      verbose_cat("  -> sampling weights  :  Argument 'weights' was a data frame. Converted to a vector. (",color_codes$red_start,"WARNING",color_codes$red_end,")\n", verbose = verbose)
    }
    verbose_cat("  -> sampling weights  : ", length(weights), " values (",color_codes$yellow_start,"note that not all explanations handle weights",color_codes$yellow_end,")\n", verbose = verbose)
    if (length(weights) != nrow(data)) {
      verbose_cat("  -> sampling weights  :  length of 'weights' is different than number of rows in 'data' (",color_codes$red_start,"WARNING",color_codes$red_end,") \n", verbose = verbose)
    }
  }

  if (is.null(model_info)) {
    # extract defaults
    model_info <- model_info(model)
    verbose_cat("  -> model_info        :  package", model_info$package[1], ", ver.", model_info$ver[1], ", task", model_info$type, "(", color_codes$yellow_start,"default",color_codes$yellow_end, ")", "\n", verbose = verbose)
  } else {
    verbose_cat("  -> model_info        :  package", model_info$package[1], ", ver.", model_info$ver[1], ", task", model_info$type, "\n", verbose = verbose)
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
      verbose_cat("  -> predict function  : yhat.default will be used (",color_codes$yellow_start,"default",color_codes$yellow_end,")\n", verbose = verbose)
    } else {
      verbose_cat("  -> predict function  : ",matching_yhat[1]," will be used (",color_codes$yellow_start,"default",color_codes$yellow_end,")\n", verbose = verbose)
      if (model_info$type == "classification") {
        verbose_cat("  -> predict function  : ",matching_yhat[1]," may treat 1st class as significant in case of multilabel target. (",color_codes$yellow_start,"default",color_codes$yellow_end,")\n", verbose = verbose)
      }
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
      verbose_cat("  -> predicted values  :  the predict_function returns an error when executed (",color_codes$red_start,"WARNING",color_codes$red_end,") \n", verbose = verbose)
    } else {
      if ((is.factor(y_hat) | is.character(y_hat))) {
        verbose_cat("  -> predicted values  :  factor (",color_codes$red_start,"WARNING",color_codes$red_end,") with levels: ", paste(unique(y_hat), collapse = ", "), "\n", verbose = verbose)
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
      verbose_cat("  -> residual function :  difference between y and yhat (",color_codes$yellow_start,"default",color_codes$yellow_end,")\n", verbose = verbose)
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
      verbose_cat("  -> residuals         :  the residual_function returns an error when executed (",color_codes$red_start,"WARNING",color_codes$red_end,") \n", verbose = verbose)
    } else {
      if ((is.factor(residuals) | is.character(residuals))) {
        verbose_cat("  -> residuals         :  factor (",color_codes$red_start,"WARNING",color_codes$red_end,") with levels: ", paste(unique(residuals), collapse = ", "), "\n", verbose = verbose)
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
                    label = label,
                    model_info = model_info,
                    weights = weights)
  explainer <- c(explainer, list(...))
  class(explainer) <- "explainer"

  # everything went OK
  verbose_cat("",color_codes$green_start,"A new explainer has been created!",color_codes$green_end,"\n", verbose = verbose)
  explainer
}

verbose_cat <- function(..., verbose = TRUE) {
  if (verbose) {
    cat(...)
  }
}

# checks if the target variable is present in the data
is_y_in_data <- function(data, y) {
  any(apply(data, 2, function(x) {
    all(as.character(x) == as.character(y))
  }))
}

#' @rdname explain
#' @export
explain <- explain.default


#
# colors for WARNING, NOTE, DEFAULT
#
color_codes <- list(yellow_start = "\033[33m", yellow_end = "\033[39m",
                    red_start = "\033[31m", red_end = "\033[39m",
                    green_start = "\033[32m", green_end = "\033[39m")

