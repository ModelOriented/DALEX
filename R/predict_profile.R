#' Instance Level Profile as Ceteris Paribus
#'
#' This function calculated individual profiles aka Ceteris Paribus Profiles.
#' From DALEX version 1.0 this function calls the \code{\link[ingredients]{ceteris_paribus}} from the \code{ingredients} package.
#' Find information how to use this function here: \url{http://ema.drwhy.ai/ceterisParibus.html}.
#'
#' @param explainer a model to be explained, preprocessed by the \code{explain} function
#' @param new_observation a new observation for which predictions need to be explained
#' @param variables character - names of variables to be explained
#' @param ... other parameters
#' @param type character, currently only the \code{ceteris_paribus} is implemented
#' @param variable_splits_type how variable grids shall be calculated? Use "quantiles" (default) for percentiles or "uniform" to get uniform grid of points. Will be passed to `ingredients`.
#'
#' @return An object of the class \code{ceteris_paribus_explainer}.
#' It's a data frame with calculated average response.
#'
#' @references Explanatory Model Analysis. Explore, Explain, and Examine Predictive Models. \url{http://ema.drwhy.ai/}
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
#' dragon_lm_predict4 <- predict_profile(dragon_lm_explainer4,
#'                 new_observation = new_dragon,
#'                 variables = c("year_of_birth", "height", "scars"))
#' head(dragon_lm_predict4)
#' plot(dragon_lm_predict4,
#'     variables = c("year_of_birth", "height", "scars"))
#'
#' \donttest{
#' library("ranger")
#' dragon_ranger_model4 <- ranger(life_length ~ year_of_birth + height +
#'                                                weight + scars + number_of_lost_teeth,
#'                                  data = dragons, num.trees = 50)
#' dragon_ranger_explainer4 <- explain(dragon_ranger_model4, data = dragons, y = dragons$year_of_birth,
#'                                 label = "model_ranger")
#' dragon_ranger_predict4 <- predict_profile(dragon_ranger_explainer4,
#'                                            new_observation = new_dragon,
#'                                            variables = c("year_of_birth", "height", "scars"))
#' head(dragon_ranger_predict4)
#' plot(dragon_ranger_predict4,
#'     variables = c("year_of_birth", "height", "scars"))
#'  }
#'

#' @name predict_profile
#' @export
predict_profile <-  function(explainer, new_observation, variables = NULL, ..., type = "ceteris_paribus", variable_splits_type = "uniform") {
  # run checks against the explainer objects
  test_explainer(explainer, has_data = TRUE, function_name = "predict_profile")
  if (type != "ceteris_paribus") stop("Currently only ceteris_paribus profiles are implemented")

  # call the ceteris_paribus from ingredients
  res <- ingredients::ceteris_paribus(explainer,
                               new_observation = new_observation,
                               variables = variables,
                               variable_splits_type = variable_splits_type,
                                ...)
  class(res) <- c("predict_profile", class(res))
  res
}

#' @name predict_profile
#' @export
individual_profile <- predict_profile
