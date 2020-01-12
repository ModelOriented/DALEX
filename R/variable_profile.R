#' Instance Level Variable Effect as Ceteris Paribus Profile
#'
#' From DALEX version 1.0 this function calls the \code{\link[ingredients]{ceteris_paribus}} from the \code{ingredients} package.
#' Find information how to use this function here: \url{https://pbiecek.github.io/PM_VEE/ceterisParibus.html}.
#'
#' @param explainer a model to be explained, preprocessed by the 'explain' function
#' @param new_observation a new observarvation for which predictions need to be explained
#' @param variables character - name of variables to be explained
#' @param ... other parameters
#'
#' @return An object of the class 'ceteris_paribus_explainer'.
#' It's a data frame with calculated average response.
#'
#' @references Predictive Models: Explore, Explain, and Debug. Human-Centered Interpretable Machine Learning \url{https://pbiecek.github.io/PM_VEE/}
#' @export
#'
#' @examples
#' new_dragon <- data.frame(year_of_birth = 200,
#'      height = 80,
#'      weight = 12.5,
#'      scars = 0,
#'      number_of_lost_teeth  = 5)
#'
#' dragon_lm_model4 <- lm(life_length ~ year_of_birth + height +
#'                                      weight + scars + number_of_lost_teeth,
#'                        data = dragons)
#' dragon_lm_explainer4 <- explain(dragon_lm_model4, data = dragons, y = dragons$year_of_birth,
#'                                 label = "model_4v")
#' dragon_lm_predict4 <- variable_profile(dragon_lm_explainer4,
#'                 new_observation = new_dragon,
#'                 variables = c("year_of_birth", "height", "scars"))
#' head(dragon_lm_predict4)
#' plot(dragon_lm_predict4)
#'
#' \dontrun{
#' library("randomForest")
#' dragon_rf_model4 <- randomForest(life_length ~ year_of_birth + height +
#'                                                weight + scars + number_of_lost_teeth,
#'                                  data = dragons)
#' dragon_rf_explainer4 <- explain(dragon_rf_model4, data = dragons, y = dragons$year_of_birth,
#'                                 label = "model_rf")
#' dragon_rf_predict4 <- variable_profile(dragon_rf_explainer4,
#'                 new_observation = new_dragon,
#'                 variables = c("year_of_birth", "height", "scars"))
#' head(dragon_rf_predict4)
#' plot(dragon_rf_predict4)
#'  }
#'

#' @name variable_profile
#' @export
variable_profile <-  function(explainer, new_observation, variables, ...) {
  # run checks against the explainer objects
  if (!("explainer" %in% class(explainer))) stop("The variable_profile() function requires an object created with explain() function.")
  if (is.null(explainer$data)) stop("The variable_profile() function requires explainers created with specified 'data' parameter.")

  # call the shap from iBreakDown
  ingredients::ceteris_paribus(explainer,
                               new_observation = new_observation,
                               variables = variables,
                                ...)
}


