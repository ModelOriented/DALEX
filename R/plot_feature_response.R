#' Plot Marginal Model Explanations (Single Variable Responses)
#'
#' Function 'plot.variable_response_explainer' plots marginal responses for one or more explainers.
#'
#' @param x a single variable exlainer produced with the 'single_feature' function
#' @param ... other explainers that shall be plotted together
#' @param use_facets logical. If TRUE then separate models are on different facets
#'
#' @return a ggplot2 object
#' @export
#' @import ggplot2
#' @importFrom grDevices dev.off pdf
#'
#' @examples
#'
#' HR_glm_model <- glm(status == "fired" ~., data = HR, family = "binomial")
#' explainer_glm <- explain(HR_glm_model, data = HR)
#' expl_glm <- feature_response(explainer_glm, "hours", "pdp")
#' head(expl_glm)
#' plot(expl_glm)
#'
#'  \dontrun{
#' library("randomForest")
#' HR_rf_model <- randomForest(as.factor(status == "fired" )~., data = HR, ntree = 100)
#' explainer_rf  <- explain(HR_rf_model, data = HR)
#' expl_rf  <- feature_response(explainer_rf, feature = "hours",
#'                        type = "pdp")
#' head(expl_rf)
#' plot(expl_rf)
#'
#' plot(expl_rf, expl_glm)
#' plot(expl_rf, expl_glm, use_facets = TRUE)
#'  }
#'

plot.feature_response_explainer <- function(x, ..., use_facets = FALSE) {
  if ("factorMerger_list" %in% class(x)) {
  }
  if ("factorMerger" %in% class(x)) {
    return(plot.variable_response_factor_explainer(x, ...))
  }
  if ("data.frame" %in% class(x)) {
    return(plot.variable_response_numeric_explainer(x, ..., use_facets = use_facets))
  }
}

#' @export
plot.factorMerger_list <- function(x, ...) {
  do.call(plot.variable_response_factor_explainer,x)
}


