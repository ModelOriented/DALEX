#' Print Dataset Level Model Diagnostics
#'
#' Generic function
#'
#' @param x an object with dataset level residual diagnostics created with \code{\link{model_diagnostics}} function
#' @param ... other parameters
#'
#' @export
print.model_diagnostics_explainer <- function(x, ...) {
  class(x) = "data.frame"
  print(summary(x))
}
