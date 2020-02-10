#' Plot Variable Profile Explanations
#'
#' @param x a model to be explained, preprocessed by the \code{\link{explain}} function
#' @param ... other parameters
#'
#' @return An object of the class \code{variable_profile_explainer}.
#'
#' @export
plot.individual_profile_explainer <- function(x, ...) {
  class(x) <- c("ceteris_paribus_explainer", "data.frame")
  plot(x, ...) +
    ingredients::show_observations(x, size = 3, ...)
}
