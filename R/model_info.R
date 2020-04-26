#' Exract info from model
#'
#' This generic function let user extract base information about model. The function returns a named list of class \code{model_info} that
#' contain about package of model, version and task type. For wrappers like \code{mlr} or \code{caret} both, package and wrapper inforamtion
#' are stored
#'
#' @param model - model object
#' @param ... - another arguments
#'
#' Currently supported packages are:
#' \itemize{
#' \item class \code{cv.glmnet} and \code{glmnet} - models created with \pkg{glmnet} package
#' \item class \code{glm} - generalized linear models
#' \item class \code{lrm} - models created with \pkg{rms} package,
#' \item class \code{model_fit} - models created with \pkg{parsnip} package
#' \item class \code{lm} - linear models created with \code{stats::lm}
#' \item class \code{ranger} - models created with \pkg{ranger} package
#' \item class \code{randomForest} - random forest models created with \pkg{randomForest} package
#' \item class \code{svm} - support vector machines models created with the \pkg{e1071} package
#' \item class \code{train} - models created with \pkg{caret} package
#' \item class \code{gbm} - models created with \pkg{gbm} package
#' }
#'
#' @return A named list of class \code{model_info}
#'
#' @rdname model_info
#' @export
#'
#' @examples
#' aps_lm_model4 <- lm(m2.price ~., data = apartments)
#' model_info(aps_lm_model4)
#'
#' library("ranger")
#' model_regr_rf <- ranger::ranger(m2.price~., data = apartments, num.trees = 50)
#' model_info(model_regr_rf)
#'
model_info <- function(model, ...)
  UseMethod("model_info")


#' @rdname model_info
#' @export
model_info.lm <- function(model, ...) {
  type <- "regression"
  package <- "stats"
  ver <- as.character(utils::packageVersion("stats"))
  model_info <- list(package = package, ver = ver, type = type)
  class(model_info) <- "model_info"
  model_info
}

#' @rdname model_info
#' @export
model_info.randomForest <- function(model, ...) {
  if (model$type == "classification" & !is.null(list(...)$task_subtype)) {
    if (list(...)$task_subtype){
      type <- "multilclass classification"
    } else {
      type <- "binary classification"
    }
  } else if (model$type == "classification" & is.null(list(...)$task_subtype)) {
    type <- "classification"
  } else {
    type <- "regression"
  }
  package <- "randomForest"
  ver <- get_pkg_ver_safe(package)
  model_info <- list(package = package, ver = ver, type = type)
  class(model_info) <- "model_info"
  model_info
}

#' @rdname model_info
#' @export
model_info.svm <- function(model, ...) {
  if (model$type == 0 & !is.null(list(...)$task_subtype)) {
    if (list(...)$task_subtype){
      type <- "multilclass classification"
    } else {
      type <- "binary classification"
    }
  } else if (model$type == 0 & is.null(list(...)$task_subtype)) {
    type <- "classification"
  } else {
    type <- "regression"
  }
  package <- "e1071"
  ver <- get_pkg_ver_safe(package)
  model_info <- list(package = package, ver = ver, type = type)
  class(model_info) <- "model_info"
  model_info
}

#' @rdname model_info
#' @export
model_info.glm <- function(model, ...) {
  if (model$family$family == "binomial") {
    type <- "binary classification"
  } else {
    type <- "regression"
  }
  package <- "stats"
  ver <- get_pkg_ver_safe(package)
  model_info <- list(package = package, ver = ver, type = type)
  class(model_info) <- "model_info"
  model_info
}

#' @rdname model_info
#' @export
model_info.lrm <- function(model, ...) {
  if (!is.null(list(...)$task_subtype)) {
    if (list(...)$task_subtype){
      type <- "multilclass classification"
    } else {
      type <- "binary classification"
    }
  } else {
    type <- "classification"
  }
  package <- "rms"
  ver <- get_pkg_ver_safe(package)
  model_info <- list(package = package, ver = ver, type = type)
  class(model_info) <- "model_info"
  model_info
}

#' @rdname model_info
#' @export
model_info.glmnet <- function(model, ...) {
  if (!is.null(model$classnames) & !is.null(list(...)$task_subtype)) {
    if (list(...)$task_subtype){
      type <- "multilclass classification"
    } else {
      type <- "binary classification"
    }
  } else if (!is.null(model$classnames) & is.null(list(...)$task_subtype)) {
    type <- "classification"
  } else {
    type <- "regression"
  }
  package <- "glmnet"
  ver <- get_pkg_ver_safe(package)
  model_info <- list(package = package, ver = ver, type = type)
  class(model_info) <- "model_info"
  model_info
}

#' @rdname model_info
#' @export
model_info.cv.glmnet <- function(model, ...) {
  if (!is.null(model$glmnet.fit$classnames) & !is.null(list(...)$task_subtype)) {
    if (list(...)$task_subtype){
      type <- "multilclass classification"
    } else {
      type <- "binary classification"
    }
  } else if (!is.null(model$glmnet.fit$classnames) & is.null(list(...)$task_subtype)) {
    type <- "classification"
  } else {
    type <- "regression"
  }
  package <- "glmnet"
  ver <- get_pkg_ver_safe(package)
  model_info <- list(package = package, ver = ver, type = type)
  class(model_info) <- "model_info"
  model_info
}

#' @rdname model_info
#' @export
model_info.ranger <- function(model, ...) {
  if (model$treetype == "Regression") {
    type <- "regression"
  } else if (!is.null(list(...)$task_subtype)) {
    if (list(...)$task_subtype){
      type <- "multilclass classification"
    } else {
      type <- "binary classification"
    }
  } else {
    type <- "classification"
  }
  package <- "ranger"
  ver <- get_pkg_ver_safe(package)
  model_info <- list(package = package, ver = ver, type = type)
  class(model_info) <- "model_info"
  model_info
}

#' @rdname model_info
#' @export
model_info.gbm <- function(model, ...) {
  if (model$distribution == "multinomial") {
    type <- "multilclass classification"
  } else if (model$distribution == "bernoulli") {
    type <- "binary classification"
  } else {
    type <- "regression"
  }
  package <- "gbm"
  ver <- get_pkg_ver_safe(package)
  model_info <- list(package = package, ver = ver, type = type)
  class(model_info) <- "model_info"
  model_info
}


#' @rdname model_info
#' @export
model_info.model_fit <- function(model, ...) {
  if (model$spec$mode == "classification" & !is.null(list(...)$task_subtype)) {
    if (list(...)$task_subtype){
      type <- "multilclass classification"
    } else {
      type <- "binary classification"
    }
  } else if (model$spec$mode == "classification" & is.null(list(...)$task_subtype)) {
    type <- "classification"
  } else {
    type <- "regression"
  }
  package_wrapper <- "parsnip"
  ver_wrapper <- get_pkg_ver_safe(package_wrapper)
  package <- model$spec$method$libs
  ver <- get_pkg_ver_safe(package)
  model_info <- list(package = c(wrapper = package_wrapper, package = package), ver = c(wrapper = ver_wrapper, package = ver), type = type)
  class(model_info) <- "model_info"
  model_info
}

#' @rdname model_info
#' @export
model_info.train <- function(model, ...) {
  if (model$modelType == "Classification" & !is.null(list(...)$task_subtype)) {
    if (list(...)$task_subtype){
      type <- "multilclass classification"
    } else {
      type <- "binary classification"
    }
  } else if (model$modelType == "Classification" & is.null(list(...)$task_subtype)) {
    type <- "classification"
  } else {
    type <- "regression"
  }
  package_wrapper <- "caret"
  ver_wrapper <- get_pkg_ver_safe(package_wrapper)
  package <- model$modelInfo$library
  if (is.null(package)) {
    package <- "stats"
  }
  ver <- get_pkg_ver_safe(package)
  model_info <- list(package = c(wrapper = package_wrapper, package = package), ver = c(wrapper = ver_wrapper, package = ver), type = type)
  class(model_info) <- "model_info"
  model_info
}

#' @rdname model_info
#' @export
model_info.rpart <- function(model, ...) {
  if (attr(model$terms, "dataClasses")[1] == "factor" & !is.null(list(...)$task_subtype)) {
    if (list(...)$task_subtype){
      type <- "multilclass classification"
    } else {
      type <- "binary classification"
    }
  } else if (attr(model$terms, "dataClasses")[1] == "factor" & is.null(list(...)$task_subtype)) {
    type <- "classification"
  } else {
    type <- "regression"
  }
  package <- "rpart"
  ver <- get_pkg_ver_safe(package)
  model_info <- list(package = package, ver = ver, type = type)
  class(model_info) <- "model_info"
  model_info
}

#' @rdname model_info
#' @export
model_info.default <- function(model, ...) {
  type <- "regression"
  package <- paste("Model of class:", class(model), "package unrecognized")
  ver <- "Unknown"
  model_info <- list(package = package, ver = ver, type = type)
  class(model_info) <- "model_info"
  model_info
}

#' Print model_info
#'
#' Function prints object of class \code{model_info} created with \code{\link{model_info}}
#'
#' @param x - an object of class \code{model_info}
#' @param ... - other parameters
#' @rdname print.model_info
#' @export
print.model_info <- function(x, ...) {
 if (length(x$package) == 2) {
   cat(paste("Wrapper package:", x$package["wrapper"], "\n"))
   cat(paste("Wrapper package version:", x$ver["wrapper"], "\n"))
   cat(paste("Package:", x$package["package"], "\n"))
   cat(paste("Package version:", x$ver["package"], "\n"))
 } else {
   cat(paste("Package:", x$package, "\n"))
   cat(paste("Package version:", x$ver, "\n"))
 }
  cat(paste("Task type:", x$type))
}

get_pkg_ver_safe <- function(package) {
  ver <- try(as.character(utils::packageVersion(package)), silent = TRUE)
  if (class(ver) == "try-error") {
    ver <- "Unknown"
  }
  ver
}
