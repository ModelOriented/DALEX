#' Plot Variable Profile Explanations
#'
#' @param x an object of the class \code{predict_profile}
#' @param ... other parameters
#'
#' @return An object of the class \code{ggplot}.
#'
#' @export
plot.predict_profile <- function(x, ...) {
  class(x) <- c("ceteris_paribus_explainer", "data.frame")
  plot(x, ...) +
    ingredients::show_observations(x, size = 3, ...)
}
