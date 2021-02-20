#' Instance Level Parts of the Model Predictions
#'
#' Instance Level Variable Attributions as Break Down, SHAP or Oscillations explanations.
#' Model prediction is decomposed into parts that are attributed for particular variables.
#' From DALEX version 1.0 this function calls the \code{\link[iBreakDown]{break_down}} or
#' \code{\link[iBreakDown:break_down_uncertainty]{shap}} functions from the \code{iBreakDown} package or
#' \code{\link[ingredients:ceteris_paribus]{ceteris_paribus}} from the \code{ingredients} package.
#' Find information how to use the \code{break_down} method here: \url{http://ema.drwhy.ai/breakDown.html}.
#' Find information how to use the \code{shap} method here: \url{http://ema.drwhy.ai/shapley.html}.
#' Find information how to use the \code{oscillations} method here: \url{http://ema.drwhy.ai/ceterisParibusOscillations.html}.
#'
#' @param explainer a model to be explained, preprocessed by the \code{explain} function
#' @param new_observation a new observation for which predictions need to be explained
#' @param ... other parameters that will be passed to \code{iBreakDown::break_down}
#' @param variable_splits named list of splits for variables. It is used by oscillations based measures. Will be passed to \code{\link[ingredients]{ceteris_paribus}}.
#' @param variables names of variables for which splits shall be calculated. Will be passed to \code{\link[ingredients]{ceteris_paribus}}.
#' @param N number of observations used for calculation of oscillations. By default 500.
#' @param variable_splits_type how variable grids shall be calculated? Will be passed to \code{\link[ingredients]{ceteris_paribus}}.
#' @param type the type of variable attributions. Either \code{shap}, \code{oscillations}, \code{oscillations_uni},
#' \code{oscillations_emp}, \code{break_down} or \code{break_down_interactions}.
#'
#' @return Depending on the \code{type} there are different classes of the resulting object.
#' It's a data frame with calculated average response.
#'
#'
#'
#' @aliases predict_parts_break_down predict_parts predict_parts_ibreak_down predict_parts_shap
#' @references Explanatory Model Analysis. Explore, Explain, and Examine Predictive Models. \url{http://ema.drwhy.ai/}
#'
#' @examples
#' library(DALEX)
#'
#' new_dragon <- data.frame(
#'     year_of_birth = 200,
#'     height = 80,
#'     weight = 12.5,
#'     scars = 0,
#'     number_of_lost_teeth  = 5
#' )
#'
#' model_lm <- lm(life_length ~ year_of_birth + height +
#'                weight + scars + number_of_lost_teeth,
#'                data = dragons)
#'
#' explainer_lm <- explain(model_lm,
#'                         data = dragons,
#'                         y = dragons$year_of_birth,
#'                         label = "model_lm")
#'
#' bd_lm <- predict_parts_break_down(explainer_lm, new_observation = new_dragon)
#' head(bd_lm)
#' plot(bd_lm)
#'
#' \donttest{
#' library("ranger")
#' model_ranger <- ranger(life_length ~ year_of_birth + height +
#'                        weight + scars + number_of_lost_teeth,
#'                        data = dragons, num.trees = 50)
#'
#' explainer_ranger <- explain(model_ranger,
#'                             data = dragons,
#'                             y = dragons$year_of_birth,
#'                             label = "model_ranger")
#'
#' bd_ranger <- predict_parts_break_down(explainer_ranger, new_observation = new_dragon)
#' head(bd_ranger)
#' plot(bd_ranger)
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
          "oscillations_uni"        = predict_parts_oscillations_uni(explainer, new_observation, ...),
          "oscillations_emp"        = predict_parts_oscillations_emp(explainer, new_observation, ...),
          stop("The type argument shall be either 'shap' or 'break_down' or 'break_down_interactions' or 'oscillations' or 'oscillations_uni' or 'oscillations_emp'")
  )
}

#' @name predict_parts
#' @export
predict_parts_oscillations <- function(explainer, new_observation, ...) {
  # run checks against the explainer objects
  test_explainer(explainer, has_data = TRUE, function_name = "predict_parts_oscillations")

  # call the ceteris_paribus
  cp <- ingredients::ceteris_paribus(explainer,
                                     new_observation = new_observation,
                                     ...)
  res <- ingredients::calculate_oscillations(cp)
  class(res) <- c('predict_parts', class(res))
  res
}

#' @name predict_parts
#' @export
predict_parts_oscillations_uni <- function(explainer, new_observation, variable_splits_type = "uniform", ...) {
  # run checks against the explainer objects
  test_explainer(explainer, has_data = TRUE, function_name = "predict_parts_oscillations_uni")

  # call the ceteris_paribus
  cp <- ingredients::ceteris_paribus(explainer,
                                     new_observation = new_observation,
                                     variable_splits_type = variable_splits_type,
                                     ...)
  res <- ingredients::calculate_oscillations(cp)
  class(res) <- c('predict_parts', class(res))
  res
}

#' @name predict_parts
#' @export
predict_parts_oscillations_emp <- function(explainer, new_observation, variable_splits = NULL, variables = colnames(explainer$data), N = 500, ...) {
  # run checks against the explainer objects
  test_explainer(explainer, has_data = TRUE, function_name = "predict_parts_oscillations_emp")
  variables <- intersect(variables, colnames(new_observation))
  N <- min(N, nrow(explainer$data))
  data_sample <- explainer$data[sample(1:nrow(explainer$data), N),]

  variable_splits <- lapply(variables, function(var) {
    data_sample[,var]
  })
  names(variable_splits) <- variables

  # call the ceteris_paribus
  cp <- ingredients::ceteris_paribus(explainer,
                                     new_observation = new_observation,
                                     variable_splits = variable_splits,
                                     variables = variables,
                                     ...)
  res <- ingredients::calculate_oscillations(cp)
  class(res) <- c('predict_parts', class(res))
  res
}

#' @name predict_parts
#' @export
predict_parts_break_down <- function(explainer, new_observation, ...) {
  # run checks against the explainer objects
  test_explainer(explainer, has_data = TRUE, function_name = "predict_parts_break_down")

  # call the break_down
  res <- iBreakDown::break_down(explainer,
                                new_observation = new_observation,
                                ...)
  class(res) <- c('predict_parts', class(res))
  res
}

#' @name predict_parts
#' @export
predict_parts_break_down_interactions <- function(explainer, new_observation, ...) {
  # run checks against the explainer objects
  test_explainer(explainer, has_data = TRUE, function_name = "predict_parts_break_down_interactions")

  # call the break_down
  res <- iBreakDown::break_down(explainer,
                                new_observation = new_observation,
                                ...,
                                interactions = TRUE)
  class(res) <- c('predict_parts', class(res))
  res
}

#' @name predict_parts
#' @export
predict_parts_shap <- function(explainer, new_observation, ...) {
  # run checks against the explainer objects
  test_explainer(explainer, has_data = TRUE, function_name = "predict_parts_shap")

  # call the shap from iBreakDown
  res <- iBreakDown::shap(explainer,
                          new_observation = new_observation,
                          ...)
  class(res) <- c('predict_parts', class(res))
  res
}

#' @name predict_parts
#' @export
variable_attribution <- predict_parts
