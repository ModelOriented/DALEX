#' Plot Dataset Level Model Profile Explanations
#'
#' @param x a variable profile explanation, created with the \code{\link{model_profile}} function
#' @param ... other parameters
#' @param geom either \code{"aggregates"}, \code{"profiles"}, \code{"points"} determines which will be plotted
#'
#' @return An object of the class \code{ggplot}.
#'
#' \subsection{aggregates}{
#' \itemize{
#'  \item{color}{a character. Either name of a color, or hex code for a color,
#'   or \code{_label_} if models shall be colored, or \code{_ids_} if instances shall be colored}
#'  \item{size}{a numeric. Size of lines to be plotted}
#'  \item{alpha}{a numeric between \code{0} and \code{1}. Opacity of lines}
#'  \item{facet_ncol}{number of columns for the \code{\link[ggplot2]{facet_wrap}}}
#'  \item{variables}{if not \code{NULL} then only \code{variables} will be presented}
#'  \item{title}{a character. Partial and accumulated dependence explainers have deafult value.}
#'  \item{subtitle}{a character. If \code{NULL} value will be dependent on model usage.}
#' }
#' }
#'
#' @examples
#' titanic_glm_model <- glm(survived~., data = titanic_imputed, family = "binomial")
#' explainer_glm <- explain(titanic_glm_model, data = titanic_imputed)
#' expl_glm <- model_profile(explainer_glm, "fare")
#' plot(expl_glm)
#'
#'  \donttest{
#' library("ranger")
#' titanic_ranger_model <- ranger(survived~., data = titanic_imputed, num.trees = 50,
#'                                probability = TRUE)
#' explainer_ranger  <- explain(titanic_ranger_model, data = titanic_imputed)
#' expl_ranger <- model_profile(explainer_ranger)
#' plot(expl_ranger)
#' plot(expl_ranger, geom = "aggregates")
#'
#' vp_ra <- model_profile(explainer_ranger, type = "partial", variables = c("age", "fare"))
#' plot(vp_ra, variables = c("age", "fare"), geom = "points")
#'
#' vp_ra <- model_profile(explainer_ranger, type = "partial", k = 3)
#' plot(vp_ra)
#' plot(vp_ra, geom = "profiles")
#' plot(vp_ra, geom = "points")
#'
#' vp_ra <- model_profile(explainer_ranger, type = "partial", groups = "gender")
#' plot(vp_ra)
#' plot(vp_ra, geom = "profiles")
#' plot(vp_ra, geom = "points")
#'
#' vp_ra <- model_profile(explainer_ranger, type = "accumulated")
#' plot(vp_ra)
#' plot(vp_ra, geom = "profiles")
#' plot(vp_ra, geom = "points")
#'  }
#'
#' @export
plot.model_profile <- function(x, ..., geom = "aggregates") {
  switch(geom,
         aggregates = plot.model_profile_aggregates(x, ...),
         profiles = plot.model_profile_profiles(x, ...),
         points = plot.model_profile_points(x, ...),
         stop("Currently only geom = 'aggregates', 'profiles' or 'points' are implemented")
  )
}


plot.model_profile_aggregates <- function(x, ...) {
#  plot(x$agr_profiles, ..., color = x$color)
  # fix for https://github.com/ModelOriented/DALEX/issues/237
  tmp <- c(x = list(x), list(...), color = x$color)
  n_profiles <- sum(unlist(sapply(tmp, class)) == "model_profile")
  if (n_profiles > 1) tmp$color <- "_label_"
  tmp <- lapply(tmp, function(el) if("model_profile" %in% class(el)) el$agr_profiles else el)
  do.call(plot, tmp)
}

plot.model_profile_profiles <- function(x, ...) {
  plot(x$cp_profiles, ..., size = 0.5, color = "grey") +
    ingredients::show_aggregated_profiles(x$agr_profiles, ..., size = 2, color = x$color)
}

plot.model_profile_points <- function(x, ...) {
  plot(x$cp_profiles, ..., color = "grey", size = 0.5) +
    ingredients::show_aggregated_profiles(x$agr_profiles, ..., size = 2, color = x$color) +
    ingredients::show_observations(x$cp_profiles, ..., size = 1)
}

