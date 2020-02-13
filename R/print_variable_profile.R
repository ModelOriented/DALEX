#' Print Dataset Level Variable Profile
#'
#' Generic function
#'
#' @param x an object with instance level residual diagnostics created with \code{\link{variable_profile}} function
#' @param ... other parameters
#'
#' @export
print.variable_profile_explainer <- function(x, ...) {
  print(x$agr_profiles)
}
