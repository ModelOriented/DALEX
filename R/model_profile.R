#' Dataset Level Variable Profile as Partial Dependence or Accumulated Local Dependence Explanations
#'
#' This function calculates explanations on a dataset level set that explore model response as a function of selected variables.
#' The explanations can be calulated as Partial Dependence Profile or  Accumulated Local Dependence Profile.
#' Find information how to use this function here: \url{https://pbiecek.github.io/ema/partialDependenceProfiles.html}.
#' The \code{variable_profile} function is a copy of \code{model_profile}.
#'
#' Underneath this function calls the \code{\link[ingredients]{partial_dependence}} or
#' \code{\link[ingredients]{accumulated_dependence}} functions from the \code{ingredients} package.
#'
#' @param explainer a model to be explained, preprocessed by the \code{explain} function
#' @param ... other parameters that will be passed to \code{ingredients::aggregate_profiles}
#' @param groups a variable name that will be used for grouping.
#' By default \code{NULL} which means that no groups shall be calculated
#' @param N number of observations used for calculation of aggregated profiles. By default \code{100}.
#' @param k number of clusters for the hclust function (for clustered profiles)
#' @param center shall profiles be centered before clustering
#' @param variables character - names of variables to be explained
#' @param variable deprecated, use variables instead
#' @param type the type of variable profile. Either \code{partial}, \code{conditional} or \code{accumulated}.
#'
#' @return An object of the class \code{model_profile}.
#' It's a data frame with calculated average model responses.
#'
#' @references Explanatory Model Analysis. Explore, Explain, and Examine Predictive Models. \url{https://pbiecek.github.io/ema/}
#'
#' @name model_profile
#' @examples
#' titanic_glm_model <- glm(survived~., data = titanic_imputed, family = "binomial")
#' explainer_glm <- explain(titanic_glm_model, data = titanic_imputed)
#' model_profile_glm_fare <- model_profile(explainer_glm, "fare")
#' plot(model_profile_glm_fare)
#'
#'  \donttest{
#' library("ranger")
#' titanic_ranger_model <- ranger(survived~., data = titanic_imputed, num.trees = 50,
#'                                probability = TRUE)
#' explainer_ranger  <- explain(titanic_ranger_model, data = titanic_imputed)
#' model_profile_ranger <- model_profile(explainer_ranger)
#' plot(model_profile_ranger, geom = "profiles")
#'
#' model_profile_ranger_1 <- model_profile(explainer_ranger, type = "partial",
#'                                         variables = c("age", "fare"))
#' plot(model_profile_ranger_1 , variables = c("age", "fare"), geom = "points")
#'
#' model_profile_ranger_2  <- model_profile(explainer_ranger, type = "partial", k = 3)
#' plot(model_profile_ranger_2 , geom = "profiles")
#'
#' model_profile_ranger_3  <- model_profile(explainer_ranger, type = "partial", groups = "gender")
#' plot(model_profile_ranger_3 , geom = "profiles")
#'
#' model_profile_ranger_4  <- model_profile(explainer_ranger, type = "accumulated")
#' plot(model_profile_ranger_4 , geom = "profiles")
#'
#' # Multiple profiles
#' model_profile_ranger_fare <- model_profile(explainer_ranger, "fare")
#' plot(model_profile_ranger_fare, model_profile_glm_fare)
#'  }
#'
#' @export
model_profile <- function(explainer, variables = NULL, N = 100, ..., groups = NULL, k = NULL, center = TRUE, type = "partial") {
  # run checks against the explainer objects
  test_explainer(explainer, has_data = TRUE, function_name = "model_profile")

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
  # or for multilabel models where agr_profiles$`_label_` > 1
  color <- if (is.null(k) &
               is.null(groups) &
               (length(unique(agr_profiles$`_label_`)) == 1 )) colors_discrete_drwhy(1) else "_label_"

  structure(
    list(cp_profiles, agr_profiles, color),
    .Names = c("cp_profiles", "agr_profiles", "color"),
    class = "model_profile")

}

#' @name model_profile
#' @export
variable_profile <- model_profile

#' @name model_profile
#' @export
single_variable <- function(explainer, variable, type = "pdp",  ...) {
  # Deprecated
  if (!exists("message_partial_dependency", envir = .DALEX.env)) {
    .DALEX.env$message_partial_dependency = TRUE
    .Deprecated("DALEX::model_profile()", package = "DALEX", msg = "'single_variable()' is deprecated. Use 'DALEX::model_profile()' instead.\nFind examples and detailed introduction at: https://pbiecek.github.io/ema/")
  }

   model_profile(explainer, variables = variable, ...)
}
