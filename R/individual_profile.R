#' Instance Level Variable Effect as Ceteris Paribus Profile
#'
#' From DALEX version 1.0 this function calls the \code{\link[ingredients]{ceteris_paribus}} from the \code{ingredients} package.
#' Find information how to use this function here: \url{https://pbiecek.github.io/ema/ceterisParibus.html}.
#'
#' @param explainer a model to be explained, preprocessed by the 'explain' function
#' @param new_observation a new observarvation for which predictions need to be explained
#' @param variables character - names of variables to be explained
#' @param ... other parameters
#'
#' @return An object of the class 'ceteris_paribus_explainer'.
#' It's a data frame with calculated average response.
#'
#' @references Explanatory Model Analysis. Explore, Explain and Examine Predictive Models. \url{https://pbiecek.github.io/ema/}
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
#' dragon_lm_predict4 <- individual_profile(dragon_lm_explainer4,
#'                 new_observation = new_dragon,
#'                 variables = c("year_of_birth", "height", "scars"))
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
#' dragon_ranger_predict4 <- individual_profile(dragon_ranger_explainer4,
#'                                            new_observation = new_dragon,
#'                                            variables = c("year_of_birth", "height", "scars"))
#' head(dragon_ranger_predict4)
#' plot(dragon_ranger_predict4)
#'  }
#'

#' @name individual_profile
#' @export
individual_profile <-  function(explainer, new_observation, variables = NULL, ...) {
  # run checks against the explainer objects
  test_expaliner(explainer, has_data = TRUE, function_name = "individual_profile")

  # call the ceteris_paribus from ingredients
  res <- ingredients::ceteris_paribus(explainer,
                               new_observation = new_observation,
                               variables = variables,
                                ...)
  class(res) <- c("individual_profile_explainer", "ceteris_paribus_explainer", "data.frame")
  res
}
