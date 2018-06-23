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
#' @param explainer a model to be explained, preprocessed by the 'explain' function
#' @param variable character - name of a single variable
#' @param type character - type of the response to be calculated.
#' Currently following options are implemented: 'pdp' for Partial Dependency and 'ale' for Accumulated Local Effects
#' @param trans function - a transformation/link function that shall be applied to raw model predictions. This will be inherited from the explainer.
#' @param ... other parameters
#'
#' @return An object of the class 'svariable_response_explainer'.
#' It's a data frame with calculated average response.
#'
#' @export
#' @importFrom pdp partial
#' @importFrom ALEPlot ALEPlot
#' @importFrom factorMerger mergeFactors
#'
#' @aliases single_variable
#' @examples
#' library("breakDown")
#' logit <- function(x) exp(x)/(1+exp(x))
#'
#' HR_glm_model <- glm(left~., data = breakDown::HR_data, family = "binomial")
#' explainer_glm <- explain(HR_glm_model, data = HR_data, trans=logit)
#' expl_glm <- variable_response(explainer_glm, "satisfaction_level", "pdp")
#' expl_glm
#'
#'  \dontrun{
#' library("randomForest")
#' HR_rf_model <- randomForest(factor(left)~., data = breakDown::HR_data, ntree = 100)
#' explainer_rf  <- explain(HR_rf_model, data = HR_data,
#'                        predict_function = function(model, x)
#'                               predict(model, x, type = "prob")[,2])
#' expl_rf  <- variable_response(explainer_rf, variable = "satisfaction_level", type = "pdp",
#'                        which.class = 2, prob = TRUE)
#' expl_rf
#' plot(expl_rf)
#'  }
#'
variable_response <- function(explainer, variable, type = "pdp", trans = explainer$link, ...) {
  if (!("explainer" %in% class(explainer))) stop("The variable_response() function requires an object created with explain() function.")
  if (is.null(explainer$data)) stop("The variable_response() function requires explainers created with specified 'data' parameter.")
  if (inherits(explainer$data, "H2OFrame")) return(variable_response_h2o(explainer, variable, type, trans, ...))
  if (class(explainer$data[,variable]) == "factor" & type != "factor") {
    message(paste("Variable", variable, " is of the class factor. Type of explainer changed to 'factor'."))
    type <- "factor"
  }
  if (type == "factor" & !("factor" %in% class(explainer$data[,variable]))) {
    stop(paste("Variable", variable, " must be a factor variable"))
  }

  switch(type,
         factor = {
           lev <- levels(factor(explainer$data[,variable]))

           preds <- lapply(lev, function(cur_lev) {
             tmp <- explainer$data
             tmp[,variable] <- factor(cur_lev, levels = lev)
             data.frame(scores = trans(explainer$predict_function(explainer$model, tmp)),
                        level = cur_lev)
           })
           preds_combined <- do.call(rbind, preds)

           res <- mergeFactors(preds_combined$scores, preds_combined$level, abbreviate = FALSE)
           res$label = explainer$label
           class(res) <-  c("variable_response_explainer", "factorMerger", "gaussianFactorMerger")
           res
         },
         pdp = {
           # pdp requires predict function with only two arguments
           predictor_pdp <- function(object, newdata) mean(explainer$predict_function(object, newdata), na.rm = TRUE)

           part <- partial(explainer$model, pred.var = variable, train = explainer$data, ..., pred.fun = predictor_pdp, recursive = FALSE)
           res <- data.frame(x = part[,1], y = trans(part$yhat), var = variable, type = type, label = explainer$label)
           class(res) <- c("variable_response_explainer", "data.frame", "pdp")
           res
         },
         ale = {
           # pdp requires predict function with only two arguments
           predictor_ale <- function(X.model, newdata) explainer$predict_function(X.model, newdata)

           # need to create a temporary file to stop ALEPlot function from plotting anytihing
           tmpfn <- tempfile()
           pdf(tmpfn)
           part <- ALEPlot(X = explainer$data, X.model = explainer$model, J = variable, pred.fun = predictor_ale)
           dev.off()
           unlink(tmpfn)
           res <- data.frame(x = part$x.values, y = trans(part$f.values), var = variable, type = type, label = explainer$label)
           class(res) <- c("variable_response_explainer", "data.frame", "ale")
           res
         },
         stop("Currently only 'pdp', 'ale' and 'factor' methods are implemented"))
}

#' @export
single_variable <- variable_response
