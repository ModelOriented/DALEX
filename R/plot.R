#' Plots Marginal Responses
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
#'
#' @examples
#' library("randomForest")
#' library("breakDown")
#' logit <- function(x) exp(x)/(1+exp(x))
#'
#' HR_glm_model <- glm(left~., data = breakDown::HR_data, family = "binomial")
#' explainer_glm <- explain(HR_glm_model, data = HR_data)
#' expl_glm <- single_variable(explainer_glm, "satisfaction_level", "pdp", trans=logit)
#' plot(expl_glm)
#'
#' HR_rf_model <- randomForest(left~., data = breakDown::HR_data, ntree = 100)
#' explainer_rf  <- explain(HR_rf_model, data = HR_data)
#' expl_rf  <- single_variable(explainer_rf, variable =  "satisfaction_level", type = "pdp")
#' plot(expl_rf)
#'
#' plot(expl_rf, expl_glm)
#'
plot.single_variable_explainer <- function(x, ...) {
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

