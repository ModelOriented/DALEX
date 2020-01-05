#' Instance Level Variable Attribution as Break Down Explanations
#'
#' From DALEX version 1.0 this function calls the \code{\link[iBreakDown]{break_down}}
#' Find information how to use this function here: \url{https://pbiecek.github.io/PM_VEE/breakDown.html}.
#'
#' @param explainer a model to be explained, preprocessed by the 'explain' function
#' @param new_observation a new observarvation for which predictions need to be explained
#' @param ... other parameters that will be passed to \code{iBreakDown::break_down}
#' @param type the type of variable attributions. Either 'shap', 'break_down' or 'ibreak_down'.
#'
#' @return An object of the class 'single_prediction_explainer'.
#' It's a data frame with calculated average response.
#'
#' @aliases variable_attribution_break_down variable_attribution variable_attribution_ibreak_down variable_attribution_shap
#' @references Predictive Models: Explore, Explain, and Debug. Human-Centered Interpretable Machine Learning \url{https://pbiecek.github.io/PM_VEE/}
#' @export
#' @name variable_attribution
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
#' dragon_lm_predict4 <- variable_attribution_break_down(dragon_lm_explainer4,
#'                 new_observation = new_dragon)
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
#' dragon_rf_predict4 <- variable_attribution_break_down(dragon_rf_explainer4,
#'                 new_observation = new_dragon)
#' head(dragon_rf_predict4)
#' plot(dragon_rf_predict4)
#'
#' library("gbm")
#' # create a gbm model
#' model <- gbm(life_length ~ year_of_birth + height + weight + scars +
#'                            number_of_lost_teeth, data = dragons,
#'              distribution = "gaussian",
#'              n.trees = 1000,
#'              interaction.depth = 4,
#'              shrinkage = 0.01,
#'              n.minobsinnode = 10,
#'              verbose = FALSE)
#'  # make an explainer for the model
#'  explainer_gbm <- explain(model, data = dragons, predict_function =
#'          function(model, x) predict(model, x, n.trees = 1000))
#'  # create a new observation
#'  exp_sgn <- variable_attribution_break_down(explainer_gbm, new_observation = new_dragon)
#'  head(exp_sgn)
#'  plot(exp_sgn)
#'  }
#'
variable_attribution_break_down <- function(explainer, new_observation, ...) {
  # run checks against the explainer objects
  if (!("explainer" %in% class(explainer))) stop("The variable_attribution_breakdown() function requires an object created with explain() function.")
  if (is.null(explainer$data)) stop("The variable_attribution_breakdown() function requires explainers created with specified 'data' parameter.")

  # call the break_down
  iBreakDown::break_down(explainer,
                         new_observation = new_observation,
                         ...)
}

#' @name variable_attribution
#' @export
variable_attribution_ibreak_down <- function(explainer, new_observation, ...) {
  # run checks against the explainer objects
  if (!("explainer" %in% class(explainer))) stop("The variable_attribution_ibreakdown() function requires an object created with explain() function.")
  if (is.null(explainer$data)) stop("The variable_attribution_ibreakdown() function requires explainers created with specified 'data' parameter.")

  # call the break_down
  iBreakDown::break_down(explainer,
                         new_observation = new_observation,
                         ...,
                         interactions = TRUE)
}

#' @name variable_attribution
#' @export
variable_attribution_shap <- function(explainer, new_observation, ...) {
  # run checks against the explainer objects
  if (!("explainer" %in% class(explainer))) stop("The variable_attribution_shap() function requires an object created with explain() function.")
  if (is.null(explainer$data)) stop("The variable_attribution_shap() function requires explainers created with specified 'data' parameter.")

  # call the shap from iBreakDown
  iBreakDown::shap(explainer,
                         new_observation = new_observation,
                         ...)
}

#' @name variable_attribution
#' @export
variable_attribution <- function(explainer, new_observation, ..., type = "break_down") {
  switch (type,
          "break_down" = variable_attribution_break_down(explainer, new_observation, ...),
          "ibreak_down" = variable_attribution_ibreak_down(explainer, new_observation, ...),
          "shap" = variable_attribution_shap(explainer, new_observation, ...),
          stop("The type argument shall be either 'shap' or 'break_down'")
  )
}

