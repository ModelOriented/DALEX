#' Instance Level Parts of the Model Predictions
#'
#' Instance Level Variable Attributions as Break Down or SHAP Explanations.
#' Model prediction is decomposed into parts that are attributed for particular variables.
#' From DALEX version 1.0 this function calls the \code{\link[iBreakDown]{break_down}} or
#' \code{\link[iBreakDown]{shap}} functions from the \code{iBreakDown} package.
#' Find information how to use this function here: \url{https://pbiecek.github.io/ema/breakDown.html}.
#'
#' @param explainer a model to be explained, preprocessed by the 'explain' function
#' @param new_observation a new observarvation for which predictions need to be explained
#' @param ... other parameters that will be passed to \code{iBreakDown::break_down}
#' @param type the type of variable attributions. Either \code{shap}, \code{oscillations}, \code{break_down} or \code{break_down_interactions}.
#'
#' @return Depending on the \code{type} there are different classess of the resulting object.
#' It's a data frame with calculated average response.
#'
#' @aliases predict_parts_break_down predict_parts predict_parts_ibreak_down predict_parts_shap
#' @references Explanatory Model Analysis. Explore, Explain and Examine Predictive Models. \url{https://pbiecek.github.io/ema/}
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
#' dragon_lm_predict4 <- predict_parts_break_down(dragon_lm_explainer4,
#'                 new_observation = new_dragon)
#' head(dragon_lm_predict4)
#' plot(dragon_lm_predict4)
#'
#' \dontrun{
#' library("ranger")
#' dragon_ranger_model4 <- ranger(life_length ~ year_of_birth + height +
#'                                                weight + scars + number_of_lost_teeth,
#'                                  data = dragons, num.trees = 50)
#' dragon_ranger_explainer4 <- explain(dragon_ranger_model4, data = dragons, y = dragons$year_of_birth,
#'                                 label = "model_ranger")
#' dragon_ranger_predict4 <- predict_parts_break_down(dragon_ranger_explainer4,
#'                                                           new_observation = new_dragon)
#' head(dragon_ranger_predict4)
#' plot(dragon_ranger_predict4)
#'}
#'
#' @name predict_parts
#' @export
predict_parts <- function(explainer, new_observation, ..., type = "break_down") {
  switch (type,
          "break_down"              = predict_parts_break_down(explainer, new_observation, ...),
          "break_down_interactions" = predict_parts_break_down_interactions(explainer, new_observation, ...),
          "shap"                    = predict_parts_shap(explainer, new_observation, ...),
          "oscillations"            = predict_parts_oscillations(explainer, new_observation, ...),
          stop("The type argument shall be either 'shap' or 'break_down' or 'break_down_interactions' or 'oscillations'")
  )
}

#' @name predict_parts
#' @export
predict_parts_oscillations <- function(explainer, new_observation, ...) {
  # run checks against the explainer objects
  test_expaliner(explainer, has_data = TRUE, function_name = "predict_parts_oscillations")

  # call the ceteris_paribus
  cp <- ingredients::ceteris_paribus(explainer,
                         new_observation = new_observation,
                         ...)
  ingredients::calculate_oscillations(cp)
}

#' @name predict_parts
#' @export
predict_parts_break_down <- function(explainer, new_observation, ...) {
  # run checks against the explainer objects
  test_expaliner(explainer, has_data = TRUE, function_name = "predict_parts_break_down")

  # call the break_down
  iBreakDown::break_down(explainer,
                         new_observation = new_observation,
                         ...)
}

#' @name predict_parts
#' @export
predict_parts_break_down_interactions <- function(explainer, new_observation, ...) {
  # run checks against the explainer objects
  test_expaliner(explainer, has_data = TRUE, function_name = "predict_parts_break_down_interactions")

  # call the break_down
  iBreakDown::break_down(explainer,
                         new_observation = new_observation,
                         ...,
                         interactions = TRUE)
}

#' @name predict_parts
#' @export
predict_parts_shap <- function(explainer, new_observation, ...) {
  # run checks against the explainer objects
  test_expaliner(explainer, has_data = TRUE, function_name = "predict_parts_shap")

  # call the shap from iBreakDown
  iBreakDown::shap(explainer,
                         new_observation = new_observation,
                         ...)
}

#' @name predict_parts
#' @export
variable_attribution <- predict_parts
