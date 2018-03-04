#' Plots Marginal Model Explanations (Single Variable Responses)
#'
#' Function 'plot.single_variable_explainer' plots marginal responses for one or more explainers.
#'
#' @param x a single variable exlainer produced with the 'single_variable' function
#' @param ... other explainers that shall be plotted together
#'
#' @return a ggplot2 object
#' @export
#' @import ggplot2
#' @importFrom grDevices dev.off pdf
#' @importFrom ggpubr ggarrange
#'
#' @examples
#' library("breakDown")
#' logit <- function(x) exp(x)/(1+exp(x))
#'
#' HR_glm_model <- glm(left~., data = breakDown::HR_data, family = "binomial")
#' explainer_glm <- explain(HR_glm_model, data = HR_data)
#' expl_glm <- single_variable(explainer_glm, "satisfaction_level", "pdp", trans=logit)
#' plot(expl_glm)
#'
#' #\dontrun{
#' library("randomForest")
#' HR_rf_model <- randomForest(factor(left)~., data = breakDown::HR_data, ntree = 100)
#' explainer_rf  <- explain(HR_rf_model, data = HR_data,
#'                        predict_function = function(model, x) predict(model, x, type = "prob")[,2])
#' expl_rf  <- single_variable(explainer_rf, variable =  "satisfaction_level", type = "pdp", which.class = 2, prob = TRUE)
#' plot(expl_rf)
#'
#' plot(expl_rf, expl_glm)
#'
#' # Example for factor variable (with factorMerger)
#' library("randomForest")
#' expl_rf  <- single_variable(explainer_rf, variable =  "sales", type = "factor")
#' plot(expl_rf)
#'
#' expl_glm  <- single_variable(explainer_glm, variable =  "sales", type = "factor", trans = logit)
#' plot(expl_glm)
#'
#' # both models
#' plot(expl_rf, expl_glm)
#' #}
#'

plot.single_variable_explainer <- function(x, ...) {
  if ("factorMerger" %in% class(x)) {
    return(plot.single_variable_factor_explainer(x, ...))
  }
  if ("data.frame" %in% class(x)) {
    return(plot.single_variable_numeric_explainer(x, ...))
  }
}

plot.single_variable_factor_explainer <- function(x, ...) {
  fex <- c(list(x), list(...))
  clusterSplit <- list(stat = "GIC", value = 2)

  all_plots <- lapply(fex, function(fm) {
    colorsDf <- factorMerger:::getOptimalPartitionDf(fm, "GIC", 2)
    mergingPathPlot <- factorMerger:::plotTree(fm,
                                               statistic = "loglikelihood",
                                               nodesSpacing = "equidistant",
                                               title = fm$label[1],
                                               alpha = 0.5,
                                               subtitle = "",
                                               color = TRUE,
                                               colorsDf = colorsDf,
                                               markBestModel = FALSE,
                                               markStars = TRUE,
                                               clusterSplit = clusterSplit,
                                               palette = NULL,
                                               panelGrid = FALSE)
    responsePlot <- factorMerger:::plotResponse(fm, "boxplot", TRUE, clusterSplit, NULL) +
      ggtitle("Partial Group Predictions")
    ggarrange(mergingPathPlot, responsePlot,
              ncol = 2,  align = "h", widths = c(2, 1))
  })
  ggarrange(plotlist = all_plots, ncol = 1, nrow = length(all_plots)) + theme_mi2()
}

plot.single_variable_numeric_explainer <- function(x, ...) {
  df <- x
  class(df) <- "data.frame"

  dfl <- list(...)
  if (length(dfl) > 0) {
    for (resp in dfl) {
      class(resp) <- "data.frame"
      df <- rbind(df, resp)
      }
  }

  variable_name <- head(df$var, 1)
  ggplot(df, aes_string(x = "x", y = "y", color = "label", shape = "type")) +
    geom_point() +
    geom_line() +
    theme_mi2() +
    scale_color_brewer(name = "Model", type = "qual", palette = "Dark2") +
    scale_shape_discrete(name = "Type") +
    ggtitle("Single variable conditional responses") +
    xlab(variable_name) + ylab(expression(hat("y")))

}

