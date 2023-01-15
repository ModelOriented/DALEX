#' Plot Instance Level Residual Diagnostics
#'
#' @param x an object with instance level residual diagnostics created with \code{\link{predict_diagnostics}} function
#' @param ... other parameters that will be passed to \code{plot.ceteris_paribus_explaine}.
#'
#' @return an \code{ggplot2} object of the class \code{gg}.
#'
#' @examples
#' \donttest{
#' library("ranger")
#' titanic_glm_model <- ranger(survived ~ gender + age + class + fare + sibsp + parch,
#'                      data = titanic_imputed)
#' explainer_glm <- explain(titanic_glm_model,
#'                          data = titanic_imputed,
#'                          y = titanic_imputed$survived)
#' johny_d <- titanic_imputed[24, c("gender", "age", "class", "fare", "sibsp", "parch")]
#'
#' pl <- predict_diagnostics(explainer_glm, johny_d, variables = NULL)
#' plot(pl)
#'
#' pl <- predict_diagnostics(explainer_glm, johny_d,
#'                        neighbors = 10,
#'                        variables = c("age", "fare"))
#' plot(pl)
#'
#' pl <- predict_diagnostics(explainer_glm,
#'                        johny_d,
#'                        neighbors = 10,
#'                        variables = c("class", "gender"))
#' plot(pl)
#'}
#'
#' @export
plot.predict_diagnostics <- function(x, ...) {
  # if variables are not specified then gow with histogram
  if (is.null(x$variables)) {
    df <- rbind(x$histogram_neighbors, x$histogram_all)
    p.value <- x$test$p.value
    statistic <- x$test$statistic
    cut_points <- x$cut_points

    pl <- ggplot(df, aes_string("Var1", "Freq", fill = "direction")) +
      geom_col() +
      scale_y_continuous("") +
      scale_x_discrete("residuals", labels = as.character(cut_points)) +
      scale_fill_manual("", values = colors_diverging_drwhy()) +
      theme_default_dalex() + theme(legend.position = "top") +
      ggtitle("Distribution of residuals",
              paste0("Difference between distributions: D ", signif(statistic, 3),
                     " p.value ", signif(p.value, 3)))
  } else {
    cp_neighbors <- x$cp_neighbors
    cp_new_instance <- x$cp_new_instance
    variables <- x$variables

    pl <- plot(cp_neighbors, color = '#ceced9', ...) +
      ingredients::show_residuals(cp_neighbors, variables = variables) +
      ingredients::show_observations(cp_new_instance, variables = variables, size = 5) +
      ingredients::show_profiles(cp_new_instance, variables = variables, size = 2) +
      ggtitle("Local stability plot")
  }
  pl
}
