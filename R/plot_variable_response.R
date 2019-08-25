#' Plot Marginal Model Explanations (Single Variable Responses)
#'
#' Function \code{\link{plot.variable_response_explainer}}
#' plots marginal responses for one or more explainers.
#'
#' @param x a single variable exlainer produced with the \code{\link{single_variable}} function
#' @param ... other explainers that shall be plotted together
#' @param use_facets logical. If TRUE then separate models are on different facets
#'
#' @return a ggplot2 object
#' @export
#' @import ggplot2
#' @importFrom grDevices dev.off pdf
#'
#' @examples
#' HR$evaluation <- factor(HR$evaluation)
#'
#' HR_glm_model <- glm(status == "fired"~., data = HR, family = "binomial")
#' explainer_glm <- explain(HR_glm_model, data = HR)
#' expl_glm <- variable_response(explainer_glm, "age", "pdp")
#' plot(expl_glm)
#'
#'  \dontrun{
#' library("randomForest")
#' HR_rf_model <- randomForest(status == "fired" ~., data = HR)
#' explainer_rf  <- explain(HR_rf_model, data = HR)
#' expl_rf  <- variable_response(explainer_rf, variable = "age",
#'                        type = "pdp")
#' plot(expl_rf)
#' plot(expl_rf, expl_glm)
#'
#' # Example for factor variable (with factorMerger)
#' expl_rf  <- variable_response(explainer_rf, variable =  "evaluation", type = "factor")
#' plot(expl_rf)
#'
#' expl_glm  <- variable_response(explainer_glm, variable =  "evaluation", type = "factor")
#' plot(expl_glm)
#'
#' # both models
#' plot(expl_rf, expl_glm)
#'  }
#'
plot.variable_response_explainer <- function(x, ..., use_facets = FALSE) {
  if ("factorMerger" %in% class(x)) {
    return(plot.variable_response_factor_explainer(x, ...))
  }
  if ("data.frame" %in% class(x)) {
    return(plot.variable_response_numeric_explainer(x, ..., use_facets = use_facets))
  }
}

plot.variable_response_factor_explainer <- function(x, ...) {
  fex <- c(list(x), list(...))
  clusterSplit <- list(stat = "GIC", value = 2)

  all_plots <- lapply(fex, function(fm) {
    colorsDf <- factorMerger::getOptimalPartitionDf(fm, "GIC", 2)
    mergingPathPlot <- factorMerger::plotTree(fm,
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
                         panelGrid = FALSE) +
        scale_x_continuous("", breaks = NULL)
    responsePlot <- factorMerger::plotResponse(fm, "boxplot", TRUE, clusterSplit, NULL) +
      ggtitle("Partial Group Predictions")
    ggpubr::ggarrange(mergingPathPlot,
              ncol = 1,  align = "h", widths = c(2))
  })
  ggpubr::ggarrange(plotlist = all_plots, ncol = 1, nrow = length(all_plots)) + theme_void()
}

plot.variable_response_numeric_explainer <- function(x, ..., use_facets = FALSE) {
  df <- combine_explainers(x, ...)

  variable_name <- head(df$var, 1)
  nlabels <- length(unique(df$label))
  pl <- ggplot(df, aes_string(x = "x", y = "y", color = "label")) +
    geom_point(size = 1) +
    geom_line(size = 1) +
    theme_drwhy() +
    scale_color_manual(name = "Model", values = colors_discrete_drwhy(nlabels)) +
    ggtitle("") +
    xlab(variable_name) + ylab("prediction")

  if (use_facets) {
    pl <- pl + facet_wrap(~label, ncol = 1, scales = "free_y") +
      theme(legend.position = "none")
  }

  pl

}

