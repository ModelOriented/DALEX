#' Marginal Response for a Single Variable
#'
#' Calculates the average model response as a function of a single selected variable.
#' Use the 'type' parameter to select the type of marginal response to be calculated.
#' Currently we have Partial Dependency and Accumulated Local Effects implemented.
#' Current implementation uses the 'pdp' package (Brandon M. Greenwell (2017).
#' pdp: An R Package for Constructing Partial Dependence Plots. The R Journal, 9(1), 421--436.)
#' and 'ALEPlot' (Dan Apley (2017). ALEPlot: Accumulated Local Effects Plots and Partial Dependence Plots.)
#'
#'
#' @param explainer a model to be explained, preprocessed by the 'explain' function
#' @param variable character - name of a single variable
#' @param type character - type of the response to be calculated.
#' Currently following options are implemented: 'pdp' for Partial Dependency and 'ale' for Accumulated Local Effects
#' @param trans function - a transformation/link function that shall be applied to raw model predictions
#' @param ... other parameters
#'
#' @return An object of the class 'single_variable_explainer'.
#' It's a data frame with calculated average response.
#'
#' @export
#' @importFrom pdp partial
#' @importFrom ALEPlot ALEPlot
#'
#' @examples
#' library("breakDown")
#' logit <- function(x) exp(x)/(1+exp(x))
#'
#' HR_glm_model <- glm(left~., data = breakDown::HR_data, family = "binomial")
#' explainer_glm <- explain(HR_glm_model, data = HR_data)
#' expl_glm <- single_variable(explainer_glm, "satisfaction_level", "pdp", trans=logit)
#' expl_glm
#'
#' \dontrun{
#' library("randomForest")
#' HR_rf_model <- randomForest(left~., data = breakDown::HR_data, ntree = 100)
#' explainer_rf  <- explain(HR_rf_model, data = HR_data)
#' expl_rf  <- single_variable(explainer_rf, variable =  "satisfaction_level", type = "pdp")
#' expl_rf
#' }
#'
single_variable <- function(explainer, variable, type = "pdp", trans = I, ...) {
  if (!("explainer" %in% class(explainer))) stop("The single_variable() function requires an object created with explain() function.")
  if (is.null(explainer$data)) stop("The single_variable() function requires explainers created with specified 'data' parameter.")

  switch(type,
         pdp = {
           part <- partial(explainer$model, pred.var = variable, train = explainer$data, ...)
           res <- data.frame(x = part[,1], y = trans(part$yhat), var = variable, type = type, label = explainer$label)
           class(res) <- c("single_variable_explainer", "data.frame", "pdp")
           res
         },
         ale = {
           # need to create a temporary file to stop ALEPlot function from plotting anytihing
           tmpfn <- tempfile()
           pdf(tmpfn)
           part <- ALEPlot(X = explainer$data, X.model = explainer$model, yhat, J = variable)
           dev.off()
           unlink(tmpfn)
           res <- data.frame(x = part$x.values, y = trans(part$f.values), var = variable, type = type, label = explainer$label)
           class(res) <- c("single_variable_explainer", "data.frame", "ale")
           res
         },
         stop("Currently only 'pdp' and 'ale' methods are implemented"))
}


