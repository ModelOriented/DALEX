#' Dataset Level Variable Profile as Partial Dependency or Accumulated Local Dependency Explanations
#'
#' This function calculates explanations on a dataset level set that explore model response as a function of selected variables.
#' The explanations can be calulated as Partial Dependency Profile or  Accumulated Local Dependency Profile.
#' Find information how to use this function here: \url{https://pbiecek.github.io/ema/partialDependenceProfiles.html}.
#'
#' Underneath this function calls the \code{\link[ingredients]{partial_dependency}} or
#' \code{\link[ingredients]{accumulated_dependency}} functions from the \code{ingredients} package.
#'
#' @param explainer a model to be explained, preprocessed by the 'explain' function
#' @param ... other parameters that will be passed to \code{iBreakDown::break_down}
#' @param groups a variable name that will be used for grouping.
#' By default \code{NULL} which means that no groups shall be calculated
#' @param N number of observations used for calculation of aggregated profiles. By default 100.
#' @param k number of clusters for the hclust function (for clustered profiles)
#' @param center shall profiles be centered before clustering
#' @param variables character - names of variables to be explained
#' @param type the type of variable profile Either 'partial' or 'accumulated'.
#'
#' @return An object of the class 'variable_profile_explainer'.
#' It's a data frame with calculated average model responses.
#'
#' @aliases variable_profile_partial_dependence variable_profile_accumulated_dependence
#' @references Explanatory Model Analysis. Explore, Explain and Examine Predictive Models. \url{https://pbiecek.github.io/ema/}
#'
#' @name variable_profile
#' @examples
#' titanic_glm_model <- glm(survived~., data = titanic_imputed, family = "binomial")
#' explainer_glm <- explain(titanic_glm_model, data = titanic_imputed)
#' expl_glm <- variable_profile(explainer_glm, "fare")
#' plot(expl_glm)
#'
#'  \dontrun{
#' library("ranger")
#' titanic_ranger_model <- ranger(survived~., data = titanic_imputed, num.trees = 50,
#'                                probability = TRUE)
#' explainer_ranger  <- explain(titanic_ranger_model, data = titanic_imputed)
#' expl_ranger <- variable_profile(explainer_ranger)
#' plot(expl_ranger, geom = "aggregates_profiles")
#'
#' vp_ra <- variable_profile(explainer_ranger, type = "partial", variables = c("age", "fare"))
#' plot(vp_ra, variables = c("age", "fare"), geom = "aggregates_profiles_points")
#'
#' vp_ra <- variable_profile(explainer_ranger, type = "partial", k = 3)
#' plot(vp_ra, geom = "aggregates_profiles")
#'
#' vp_ra <- variable_profile(explainer_ranger, type = "partial", groups = "gender")
#' plot(vp_ra, geom = "aggregates_profiles")
#'
#' vp_ra <- variable_profile(explainer_ranger, type = "accumulated")
#' plot(vp_ra, geom = "aggregates_profiles")
#'  }
#'
#' @export
variable_profile <- function(explainer, variables = NULL, N = 100, ..., groups = NULL, k = NULL, center = TRUE, type = "partial") {
  # run checks against the explainer objects
  test_expaliner(explainer, has_data = TRUE, function_name = "variable_profile")

  # calculate serveral ceteris profiles and call the aggregate profiles for partial dependency
  data <- explainer$data
  if (N < nrow(data)) {
    # sample N points
    ndata <- data[sample(1:nrow(data), N),]
  } else {
    ndata <- data
  }

  cp_profiles <- ingredients::ceteris_paribus(explainer,
                                              new_observation = ndata,
                                              variables = variables,
                                              ...)

  if (is.null(k)) {
    agr_profiles <- ingredients::aggregate_profiles(cp_profiles, ...,
                                                    groups = groups,
                                                    variables = variables,
                                                    type = type,
                                                    center = TRUE)
  } else {
    agr_profiles <- ingredients::cluster_profiles(cp_profiles,
                                                  k = k,
                                                  center = center,
                                                  variables = variables,
                                                  ...)
  }

  # color only for groups
  color <- if (is.null(k) & is.null(groups)) "#371ea3" else "_label_"

  structure(
    list(cp_profiles, agr_profiles, color),
    .Names = c("cp_profiles", "agr_profiles", "color"),
    class = "variable_profile_explainer")

}
