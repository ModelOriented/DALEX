#' Dataset Level Variable Effect as Partial Dependency Profile or Accumulated Local Effects
#'
#' From DALEX version 1.0 this function calls the \code{\link[ingredients]{accumulated_dependence}} or
#' \code{\link[ingredients]{partial_dependence}} from the \code{ingredients} package.
#' Find information how to use this function here: \url{https://pbiecek.github.io/ema/partialDependenceProfiles.html}.
#'
#' @param explainer a model to be explained, preprocessed by the 'explain' function
#' @param variables character - names of variables to be explained
#' @param type character - type of the response to be calculated.
#' Currently following options are implemented: 'partial_dependency' for Partial Dependency and 'accumulated_dependency' for Accumulated Local Effects
#' @param ... other parameters
#'
#' @return An object of the class 'aggregated_profiles_explainer'.
#' It's a data frame with calculated average response.
#'
#' @references Explanatory Model Analysis. Explore, Explain and Examine Predictive Models. \url{https://pbiecek.github.io/ema/}
#' @export
#'
#' @examples
#' titanic_glm_model <- glm(survived~., data = titanic_imputed, family = "binomial")
#' explainer_glm <- explain(titanic_glm_model, data = titanic_imputed)
#' expl_glm <- variable_effect(explainer_glm, "fare", "partial_dependency")
#' plot(expl_glm)
#'
#'  \dontrun{
#' library("ranger")
#' titanic_ranger_model <- ranger(survived~., data = titanic_imputed, num.trees = 50,
#'                                probability = TRUE)
#' explainer_ranger  <- explain(titanic_ranger_model, data = titanic_imputed)
#' expl_ranger  <- variable_effect(explainer_ranger, variables = "fare",
#'                             type = "partial_dependency")
#' plot(expl_ranger)
#' plot(expl_ranger, expl_glm)
#'
#' # Example for factor variable (with factorMerger)
#' expl_ranger_factor  <- variable_effect(explainer_ranger, variables =  "class")
#' plot(expl_ranger_factor)
#'  }
#'

#' @name variable_effect
#' @export
variable_effect <-  function(explainer, variables, ..., type = "partial_dependency") {
  switch (type,
          "partial_dependency" = variable_effect_partial_dependency(explainer, variables, ...),
          "accumulated_dependency" = variable_effect_accumulated_dependency(explainer, variables, ...),
          stop("The type argument shall be either 'partial_dependency' or 'accumulated_dependency'")
  )
}


#' @name variable_effect
#' @export
variable_effect_partial_dependency <-  function(explainer, variables, ...) {
  # run checks against the explainer objects
  if (!("explainer" %in% class(explainer))) stop("The variable_effect_partial_dependency() function requires an object created with explain() function.")
  if (is.null(explainer$data)) stop("The variable_effect_partial_dependency() function requires explainers created with specified 'data' parameter.")

  # call the shap from iBreakDown
  ingredients::partial_dependency(explainer,
                                  variables = variables,
                                  ...)
}


#' @name variable_effect
#' @export
variable_effect_accumulated_dependency <-  function(explainer, variables, ...) {
  # run checks against the explainer objects
  if (!("explainer" %in% class(explainer))) stop("The variable_effect_accumulated_dependency() function requires an object created with explain() function.")
  if (is.null(explainer$data)) stop("The variable_effect_accumulated_dependency() function requires explainers created with specified 'data' parameter.")

  # call the shap from iBreakDown
  ingredients::accumulated_dependency(explainer,
                                  variables = variables,
                                  ...)
}

