#' Print Dataset Level Model Profile
#'
#' Generic function
#'
#' @param x an object with dataset level profile created with \code{\link{model_profile}} function
#' @param ... other parameters
#'
#' @export
print.model_profile <- function(x, ...) {
  print(x$agr_profiles)
}
