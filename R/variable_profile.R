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
#' @param variables character - names of variables to be explained
#' @param type the type of variable profile Either 'partial_dependence' or 'accumulated_dependence'.
#'
#' @return An object of the class 'variable_profile_explainer'.
#' It's a data frame with calculated average model responses.
#'
#' @aliases variable_profile_partial_dependence variable_profile_accumulated_dependence
#' @references Explanatory Model Analysis. Explore, Explain and Examine Predictive Models. \url{https://pbiecek.github.io/ema/}
#'
#' @name variable_profile
#' @export
variable_profile <- function(explainer, variables = NULL, ..., type = "partial_dependence") {
  switch (type,
          "partial_dependence" = variable_profile_partial_dependence(explainer, variables, ...),
          "accumulated_dependence" = variable_profile_accumulated_dependence(explainer, variables, ...),
          stop("The type argument shall be either 'partial_dependence' or 'accumulated_dependence'")
  )
}

#' @name variable_profile
#' @export
variable_profile_partial_dependence <- function(explainer, variables = NULL, ...) {
  # run checks against the explainer objects
  test_expaliner(explainer, has_data = TRUE, function_name = "variable_profile_partial_dependence")

}

#' @name variable_profile
#' @export
variable_profile_accumulated_dependence <- function(explainer, variables = NULL, ...) {
  # run checks against the explainer objects
  test_expaliner(explainer, has_data = TRUE, function_name = "variable_profile_accumulated_dependence")

}
