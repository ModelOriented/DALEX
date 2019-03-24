#' Marginal Response for a Single Variable
#'
#' Calculates the average model response as a function of a single selected variable.
#' Use the 'type' parameter to select the type of marginal response to be calculated.
#' Currently for numeric variables we have Partial Dependency and Accumulated Local Effects implemented.
#' Current implementation uses the 'pdp' package (Brandon M. Greenwell (2017).
#' pdp: An R Package for Constructing Partial Dependence Plots. The R Journal, 9(1), 421--436.)
#' and 'ALEPlot' (Dan Apley (2017). ALEPlot: Accumulated Local Effects Plots and Partial Dependence Plots.)
#'
#' For factor variables we are using the 'factorMerger' package.
#' Please note that the argument \code{type} must be set to \code{'factor'} to use this method.
#'
#' @param x a model to be explained, or an explainer created with function `DALEX::explain()`.
#' @param data validation dataset, will be extracted from `x` if it's an explainer
#' @param predict_function predict function, will be extracted from `x` if it's an explainer
#' @param label name of the model. By default it's extracted from the 'class' attribute of the model
#' @param feature character - name of a single variable
#' @param type character - type of the response to be calculated.
#' Currently following options are implemented: 'pdp' for Partial Dependency and 'ale' for Accumulated Local Effects
#' @param which_class character, for multilabel classification you can restrict results to selected classes. By default `NULL` which means that all classes are considered.
#' @param ... other parameters
#'
#' @return An object of the class 'model_feature_response_explainer'.
#' It's a data frame with calculated average response.
#'
#' @export
#'
#' @examples
#' library("DALEX")
#'
#' HR_glm_model <- glm(status == "fired" ~ ., data = HR, family = "binomial")
#' explainer_glm <- explain(HR_glm_model, data = HR)
#' expl_glm <- model_feature_response(explainer_glm, "age", "pdp")
#' head(expl_glm)
#' plot(expl_glm)
#'
#'  \dontrun{
#' library("randomForest")
#' HR_rf_model <- randomForest(status ~ ., data = HR, ntree = 100)
#' explainer_rf  <- explain(HR_rf_model, data = HR)
#' expl_rf  <- model_feature_response(explainer_rf, feature = "age", type = "pdp")
#' head(expl_rf)
#' plot(expl_rf)
#'
#' expl_rf  <- model_feature_response(explainer_rf, feature = "age", type = "pdp",
#'                        which_class = 2)
#' plot(expl_rf)
#'  }
#'
#' @export
#' @rdname model_feature_response
model_feature_response <- function(x, ...)
  UseMethod("model_feature_response")

#' @export
#' @rdname model_feature_response
model_feature_response.explainer <- function(x, feature, type = "pdp", which_class = NULL, ...) {
  if (is.null(x$data)) stop("The model_feature_response() function requires explainers created with specified 'data' parameter.")
  # extracts model, data and predict function from the explainer
  model <- x$model
  data <- x$data
  predict_function <- x$predict_function
  label <- x$label

  model_feature_response.default(model, data = data,
                                 predict_function = predict_function,
                                 feature = feature, type = type,
                                 label = label, which_class = which_class, ...)
}

#' @export
#' @rdname model_feature_response
model_feature_response.default <- function(x, data, predict_function,
                                   feature, type = "pdp", label = class(x)[1], which_class = NULL, ...) {
  if (class(data[,feature]) == "factor" & type != "factor") {
    message(paste("Variable", feature, " is of the class factor. Type of explainer changed to 'factor'."))
    type <- "factor"
  }
  if (type == "factor" & !("factor" %in% class(data[,feature]))) {
    stop(paste("Variable", feature, " must be a factor feature"))
  }

  # check if prediction returns more columns (multi label classification)
  tmp <- predict_function(x, head(data, 1))
  if (length(c(tmp)) > 1) { # multi label classification
    selected_classess <- seq_along(c(tmp))
    selected_names <- colnames(tmp)
    # only some
    if (!is.null(which_class)) {
      selected_classess <- which_class
    }
    res <- lapply(selected_classess, function(which_class) {
      predict_function_1d <- function(...) predict_function(...)[,which_class]
      model_feature_response_1d(x = x, data = data, predict_function = predict_function_1d,
                                       feature = feature, type = type,
                                       label = paste0(label, ".", selected_names[which_class]), ...)
    })
    # it's a factor Merger
    if ("factorMerger" %in% class(res[[1]]) ) {
      class(res) <- "factorMerger_list"
    } else {
      res <- do.call(rbind, res)
    }
  } else {
    predict_function_1d <- predict_function
    res <- model_feature_response_1d(x = x, data = data, predict_function = predict_function_1d,
                                     feature = feature, type = type, label = label, ...)
  }
  res
}

#' @export
print.factorMerger_list <- function(x, ...) {
  invisible(lapply(x, print, ...))
}

model_feature_response_1d <- function(x, data, predict_function,
                                      feature, type = "pdp", label = class(x)[1],  ...) {
  switch(type,
         factor = {
           lev <- levels(factor(data[,feature]))

           preds <- lapply(lev, function(cur_lev) {
             tmp <- data
             tmp[,feature] <- factor(cur_lev, levels = lev)
             data.frame(scores = predict_function(x, tmp),
                        level = cur_lev)
           })
           preds_combined <- do.call(rbind, preds)

           res <- factorMerger::mergeFactors(preds_combined$scores, preds_combined$level, abbreviate = FALSE)
           res$label = label
           class(res) <-  c("model_feature_response_explainer", "factorMerger", "gaussianFactorMerger")
           res
         },
         pdp = {
           # pdp requires predict function with only two arguments
           predictor_pdp <- function(object, newdata) mean(predict_function(object, newdata), na.rm = TRUE)

           part <- pdp::partial(x, pred.var = feature, train = data, ..., pred.fun = predictor_pdp, recursive = FALSE)
           res <- data.frame(x = part[,1], y = part$yhat, var = feature, type = type, label = label)
           class(res) <- c("model_feature_response_explainer", "data.frame", "pdp")
           res
         },
         ale = {
           # pdp requires predict function with only two arguments
           predictor_ale <- function(X.model, newdata) predict_function(X.model, newdata)

           # need to create a temporary file to stop ALEPlot function from plotting anytihing
           tmpfn <- tempfile()
           pdf(tmpfn)
           part <- ALEPlot::ALEPlot(X = data, X.model = x, J = feature, pred.fun = predictor_ale)
           dev.off()
           unlink(tmpfn)
           res <- data.frame(x = part$x.values, y = part$f.values, var = feature, type = type, label = label)
           class(res) <- c("model_feature_response_explainer", "data.frame", "ale")
           res
         },
         stop("Currently only 'pdp', 'ale' and 'factor' methods are implemented"))
}
