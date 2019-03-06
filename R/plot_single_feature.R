#' Plots Marginal Model Explanations (Single Variable Responses)
#'
#' Function 'plot.variable_response_explainer' plots marginal responses for one or more explainers.
#'
#' @param x a single variable exlainer produced with the 'single_variable' function
#' @param ... other explainers that shall be plotted together
#'
#' @return a ggplot2 object
#' @export
#' @import ggplot2
#' @importFrom grDevices dev.off pdf
#' @importFrom ggpubr ggarrange
#' @importFrom factorMerger getOptimalPartitionDf plotTree plotResponse
#'
#' @examples
#' library("DALEX")
#'
#' HR_glm_model <- glm(status == "fired" ~., data = HR, family = "binomial")
#' explainer_glm <- explain(HR_glm_model, data = HR)
#' expl_glm <- model_feature_response(explainer_glm, "hours", "pdp")
#' head(expl_glm)
#' plot(expl_glm)
#'
#'  \dontrun{
#' library("randomForest")
#' HR_rf_model <- randomForest(status~., data = HR, ntree = 100)
#' explainer_rf  <- explain(HR_rf_model, data = HR)
#' expl_rf  <- model_feature_response(explainer_rf, feature = "hours",
#'                        type = "pdp")
#' head(expl_rf)
#' plot(expl_rf)
#'
#' plot(expl_rf, expl_glm)
#'
#' # Example for factor variable (with factorMerger)
#' library("randomForest")
#' expl_rf  <- model_feature_response(explainer_rf, feature =  "gender", type = "factor")
#' head(expl_rf)
#' plot(expl_rf)
#'  }
#'

plot.model_feature_response_explainer <- function(x, ...) {
  if ("factorMerger_list" %in% class(x)) {
  }
  if ("factorMerger" %in% class(x)) {
    return(plot.variable_response_factor_explainer(x, ...))
  }
  if ("data.frame" %in% class(x)) {
    return(plot.variable_response_numeric_explainer(x, ...))
  }
}

#' @export
plot.factorMerger_list <- function(x, ...) {
  do.call(plot.variable_response_factor_explainer,x)
}

plot.variable_response_factor_explainer <- function(x, ...) {
  fex <- c(list(x), list(...))
  clusterSplit <- list(stat = "GIC", value = 2)

  all_plots <- lapply(fex, function(fm) {
    colorsDf <- getOptimalPartitionDf(fm, "GIC", 2)
    mergingPathPlot <- plotTree(fm,
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
        scale_x_continuous("", breaks=NULL)
    responsePlot <- plotResponse(fm, "boxplot", TRUE, clusterSplit, NULL) +
      ggtitle("Partial Group Predictions")
    ggarrange(mergingPathPlot, responsePlot,
              ncol = 2,  align = "h", widths = c(2, 1))
  })
  ggarrange(plotlist = all_plots, ncol = 1, nrow = length(all_plots))
}

plot.variable_response_numeric_explainer <- function(x, ...) {
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
    scale_color_brewer(name = "Model", type = "qual", palette = "Dark2") +
    scale_shape_discrete(name = "Type") +
    ggtitle("Variable response") +
    xlab(variable_name) + ylab(expression(hat("y")))

}

