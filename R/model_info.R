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
#' \item class `cv.glmnet` and `glmnet` - models created with `glmnet` package
#' \item class `glm` - generalized linear models
#' \item class `model_fit` - models created with `parsnip` package
#' \item class `lm` - linear models created with `stats::lm`
#' \item class `ranger` - models created with `ranger` package
#' \item class `randomForest` - random forest models created with `randomForest` package
#' \item class `svm` - support vector machines models created with the `e1071` package
#' \item class `train` - models created with `caret` package
#' \item class `gbm` - models created with `gbm` package
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
  type <- model$type
  package <- "randomForest"
  ver <- get_pkg_ver_safe(package)
  model_info <- list(package = package, ver = ver, type = type)
  class(model_info) <- "model_info"
  model_info
}

#' @rdname model_info
#' @export
model_info.svm <- function(model, ...) {
  if (model$type == 0) {
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
    type <- "classification"
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
model_info.glmnet <- function(model, ...) {
  if (!is.null(model$classnames)) {
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
  if (!is.null(model$glmnet.fit$classnames)) {
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
  if (model$distribution == "bernoulli" || model$distribution == "multinomial") {
    type <- "classification"
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
  type <- model$spec$mode
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
  type <- model$modelType
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
  if (attr(model$terms, "dataClasses")[1] == "factor") {
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
