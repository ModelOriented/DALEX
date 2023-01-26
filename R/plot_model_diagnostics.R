#' Plot Dataset Level Model Diagnostics
#'
#' @param x a data.frame to be explained, preprocessed by the \code{\link{model_diagnostics}} function
#' @param ... other object to be included to the plot
#' @param variable character - name of the variable on OX axis to be explained, by default \code{y_hat}
#' @param yvariable character - name of the variable on OY axis, by default \code{residuals}
#' @param smooth logical shall the smooth line be added
#'
#' @return an object of the class \code{model_diagnostics_explainer}.
#'
#' @examples
#' apartments_lm_model <- lm(m2.price ~ ., data = apartments)
#' explainer_lm <- explain(apartments_lm_model,
#'                          data = apartments,
#'                          y = apartments$m2.price)
#' diag_lm <- model_diagnostics(explainer_lm)
#' diag_lm
#' plot(diag_lm)
#' \donttest{
#' library("ranger")
#' apartments_ranger_model <- ranger(m2.price ~ ., data = apartments)
#' explainer_ranger <- explain(apartments_ranger_model,
#'                          data = apartments,
#'                          y = apartments$m2.price)
#' diag_ranger <- model_diagnostics(explainer_ranger)
#' diag_ranger
#' plot(diag_ranger)
#' plot(diag_ranger, diag_lm)
#' plot(diag_ranger, diag_lm, variable = "y")
#' plot(diag_ranger, diag_lm, variable = "construction.year")
#' plot(diag_ranger, variable = "y", yvariable = "y_hat")
#'}
#' @export
plot.model_diagnostics <- function(x, ..., variable = "y_hat", yvariable = "residuals", smooth = TRUE) {
  dfl <- c(list(x), list(...))
  all_models <- do.call(rbind, dfl)
  class(all_models) <- "data.frame"
  nlabels <- length(unique(all_models$label))

   pl <- ggplot(all_models, aes_string(x = variable, y = yvariable, color = "label", group = "label")) +
    geom_point(size = 0.1) +
    theme_default_dalex() +
    scale_color_manual(name = "Model", values = colors_discrete_drwhy(nlabels))

   # add smooth
   if (smooth)
     pl <- pl + geom_smooth(se = FALSE, color = "grey")

   # add hline
   if (yvariable == "residuals")
     pl <- pl + geom_hline(yintercept = 0, color = "grey", lty = 2, size = 1)

    pl + ggtitle("Model diagnostics", paste0(variable, " against ", yvariable))
}
