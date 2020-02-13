#' Plot Variable Profile Explanations
#'
#' @param x a variable profile explanation, created with the \code{\link{variable_profile}} function
#' @param ... other parameters
#' @param geom either \code{"aggregates"}, \code{"aggregates_profiles"}, \code{"aggregates_profiles_points"} determines which will be plotted
#'
#' @return An object of the class \code{ggplot}.
#'
#' @export
#
plot.variable_profile_explainer <- function(x, ..., geom = "aggregates") {
  pl <- switch(geom,
         aggregates = plot.variable_profile_explainer_aggregates(x, ...),
         aggregates_profiles = plot.variable_profile_explainer_aggregates_profiles(x, ...),
         aggregates_profiles_points = plot.variable_profile_explainer_aggregates_profiles_points(x, ...)
  )
  pl
}



plot.variable_profile_explainer_aggregates <- function(x, ...) {
  plot(x$agr_profiles, ..., color = x$color)
}

plot.variable_profile_explainer_aggregates_profiles <- function(x, ...) {
  plot(x$cp_profiles, ..., size = 0.5, color = "grey") +
    ingredients::show_aggregated_profiles(x$agr_profiles, ..., size = 2, color = x$color)
}

plot.variable_profile_explainer_aggregates_profiles_points <- function(x, ...) {
  plot(x$cp_profiles, ..., color = "grey", size = 0.5) +
    ingredients::show_aggregated_profiles(x$agr_profiles, ..., size = 2, color = x$color) +
    ingredients::show_observations(x$cp_profiles, ..., size = 1)
}

