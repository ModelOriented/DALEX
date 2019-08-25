#' Plot Break Down Explanations (Single Prediction)
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
#'  \dontrun{
#' new_dragon <- data.frame(year_of_birth = 200,
#'      height = 80,
#'      weight = 12.5,
#'      scars = 0,
#'      number_of_lost_teeth  = 5)
#'
#' dragon_lm_model4 <- lm(life_length ~ year_of_birth + height +
#'                                      weight + scars + number_of_lost_teeth,
#'                        data = dragons)
#' dragon_lm_explainer4 <- explain(dragon_lm_model4, data = dragons, y = dragons$year_of_birth,
#'                                 label = "model_4v")
#' dragon_lm_predict4 <- prediction_breakdown(dragon_lm_explainer4, observation = new_dragon)
#' plot(dragon_lm_predict4)
#'
#' library("randomForest")
#' dragon_rf_model4 <- randomForest(life_length ~ year_of_birth + height + weight +
#'                                                scars + number_of_lost_teeth,
#'                                  data = dragons)
#' dragon_rf_explainer4 <- explain(dragon_rf_model4, data = dragons, y = dragons$year_of_birth,
#'                                 label = "model_rf")
#' dragon_rf_predict4 <- prediction_breakdown(dragon_rf_explainer4, observation = new_dragon)
#' plot(dragon_rf_predict4)
#'
#' # both models
#' plot(dragon_rf_predict4, dragon_lm_predict4)
#'
#' library("gbm")
#' # create a gbm model
#' model <- gbm(life_length ~ year_of_birth + height + weight + scars + number_of_lost_teeth,
#'              data = dragons,
#'              distribution = "gaussian",
#'              n.trees = 1000,
#'              interaction.depth = 4,
#'              shrinkage = 0.01,
#'              n.minobsinnode = 10,
#'              verbose = FALSE)
#'  # make an explainer for the model
#'  explainer_gbm <- explain(model, data = dragons, predict_function =
#'          function(model, x) predict(model, x, n.trees = 1000))
#'  # create a new observation
#'  exp_sgn <- prediction_breakdown(explainer_gbm, observation = new_dragon)
#'  head(exp_sgn)
#'  plot(exp_sgn)
#'
#'  exp_sgn <- prediction_breakdown(explainer_gbm, observation = new_dragon, baseline = 0)
#'  plot(exp_sgn)
#'  }
#'
plot.prediction_breakdown_explainer <- function(x, ..., add_contributions = TRUE,
                                                vcolors = c("-1" = "#f05a71", "0" = "#371ea3", "1" = "#8bdcbe", "X" = "#371ea3"),
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
    # zero for intercept and final profnosis
    baseline <- head(broken_cumm$contribution[broken_cumm$variable_name == "Intercept"],1)
    broken_cumm$prev[broken_cumm$variable == "(Intercept)"] = baseline
    broken_cumm$prev[broken_cumm$variable == "final_prognosis"] = baseline

    if (is.null(df)) {
      df <- broken_cumm
    } else {
      df <- rbind(df, broken_cumm)
    }
  }
  df$position <- seq_along(df$position)
  ctmp <- as.character(rounding_function(df$trans_contribution, digits))
  df$pretty_text <-
    paste0(ifelse((substr(ctmp, 1, 1) == "-") |
                    (df$variable == "final_prognosis") |
                    (df$variable == "(Intercept)"), "", "+"), ctmp)

  position <- contribution <- cummulative <- prev <- trans_contribution  <- pretty_text <- NULL
  broken_baseline <- df[df$variable_name == "Intercept",]

  pl <- ggplot(df, aes(x = position + 0.5,
                                y = pmax(cummulative, prev),
                                xmin = position + 0.1, xmax = position + 0.8,
                                ymin = cummulative, ymax = prev,
                                fill = sign,
                                label = pretty_text)) +
    geom_errorbarh(data = df[!(df$variable == "final_prognosis"),],
                   aes(xmax = position + 0.1,
                       xmin = position + 1.8,
                       y = cummulative), color = "#371ea3", height = 0) +
    geom_rect(alpha = 0.9) +
    geom_hline(data = broken_baseline, aes(yintercept = contribution), lty = 3, alpha = 0.5, color = "#371ea3") +
    facet_wrap(~label, scales = "free_y", ncol = 1)

  if (add_contributions)
    pl <- pl + geom_text(nudge_y = 0.1, vjust = 0.5, hjust = 0, color = "#371ea3")

  pl <- pl +
    scale_y_continuous(expand = c(0.1, 0.1), name = "") +
    scale_x_continuous(expand = c(0, 0), labels = df$variable, breaks = df$position + 0.5, name =  "") +
    scale_fill_manual(values = vcolors) +
    coord_flip() +
    theme_drwhy_vertical() + theme(legend.position = "none")

  pl
}

