#' Plots Local Explanations (Single Prediction)
#'
#' Function 'plot.single_prediction_explainer' plots break down plots for a single prediction.
#'
#' @param x a single prediction exlainer produced with the 'single_prediction' function
#' @param ... other explainers that shall be plotted together
#' @param add_contributions shall variable contributions to be added on plot?
#' @param vcolors named vector with colors
#' @param digits number of decimal places (round) or significant digits (signif) to be used.
#' See the \code{rounding_function} argument
#' @param rounding_function function that is to used for rounding numbers.
#' It may be \code{signif()} which keeps a specified number of significant digits.
#' Or the default \code{round()} to have the same precision for all components
#'
#' @return a ggplot2 object
#' @export
#' @import ggplot2
#'
#' @examples
#' library("randomForest")
#' library("breakDown")
#' new.wine <- data.frame(citric.acid = 0.35,
#'      sulphates = 0.6,
#'      alcohol = 12.5,
#'      pH = 3.36,
#'      residual.sugar = 4.8)
#'
#' wine_lm_model4 <- lm(quality ~ pH + residual.sugar + sulphates + alcohol, data = wine)
#' wine_lm_explainer4 <- explain(wine_lm_model4, data = wine, label = "model_4v")
#' wine_lm_predict4 <- single_prediction(wine_lm_explainer4, observation = new.wine)
#' plot(wine_lm_predict4)
#'
#' wine_rf_model4 <- randomForest(quality ~ pH + residual.sugar + sulphates + alcohol, data = wine)
#' wine_rf_explainer4 <- explain(wine_rf_model4, data = wine, label = "model_rf")
#' wine_rf_predict4 <- single_prediction(wine_rf_explainer4, observation = new.wine)
#' plot(wine_rf_predict4)
#'
#' # both models
#' plot(wine_rf_predict4, wine_lm_predict4)
#'
#' library("gbm")
#' # create a gbm model
#' model <- gbm(quality ~ pH + residual.sugar + sulphates + alcohol, data = wine,
#'              distribution = "gaussian",
#'              n.trees = 1000,
#'              interaction.depth = 4,
#'              shrinkage = 0.01,
#'              n.minobsinnode = 10,
#'              verbose = FALSE)
#'  # make an explainer for the model
#'  explainer_gbm <- explain(model, data = wine)
#'  # create a new observation
#'  exp_sgn <- single_prediction(explainer_gbm, observation = new.wine,
#'               n.trees = 1000)
#'  exp_sgn
#'  plot(exp_sgn)
#'  plot(wine_rf_predict4, wine_lm_predict4, exp_sgn)
#'
#'  # session info
#'  sessionInfo()
#'
plot.single_prediction_explainer <- function(x, ..., add_contributions = TRUE,
                                             vcolors = c("-1" = "#d8b365", "0" = "#f5f5f5", "1" = "#5ab4ac", "X" = "darkgrey"),
                                             digits = 3, rounding_function = round) {
  df <- NULL
  dfl <- c(list(x), list(...))
  for (broken_cumm in dfl) {
    constant <- attr(broken_cumm, "baseline")
    broken_cumm$variable <- as.character(broken_cumm$variable)
    broken_cumm$variable_name <- as.character(broken_cumm$variable_name)
    broken_cumm$prev <- constant + broken_cumm$cummulative - broken_cumm$contribution
    broken_cumm$cummulative <- constant + broken_cumm$cummulative
    broken_cumm$trans_contribution <- broken_cumm$cummulative - broken_cumm$prev
    if (is.null(df)) {
      df <- broken_cumm
    } else {
      df <- rbind(df, broken_cumm)
    }
  }
  df$position <- seq_along(df$position)

  position <- cummulative <- prev <- trans_contribution <- NULL

  pl <- ggplot(df, aes(x = position + 0.5,
                                y = pmax(cummulative, prev),
                                xmin = position, xmax = position + 0.95,
                                ymin = cummulative, ymax = prev,
                                fill = sign,
                                label = sapply(trans_contribution, function(tmp) as.character(rounding_function(tmp, digits))))) +
    geom_errorbarh(data = df[!(df$variable == "final_prognosis"),],
                   aes(xmax = position,
                       xmin = position + 2,
                       y = cummulative), height = 0,
                   lty = "F2") +
    geom_rect(alpha = 0.9) +
    geom_hline(yintercept = constant) +
    facet_wrap(~label, scales = "free_y", ncol = 1)

  if (add_contributions)
    pl <- pl + geom_text(nudge_y = 0.1, vjust = 0.5, hjust = 0)

  pl <- pl +
    scale_y_continuous(expand = c(0.1, 0.1), name = "") +
    scale_x_continuous(labels = df$variable, breaks = df$position + 0.5, name =  "") +
    scale_fill_manual(values = vcolors) +
    coord_flip() +
    theme_mi2() + theme(legend.position = "none", panel.border = element_blank())

  pl
}

