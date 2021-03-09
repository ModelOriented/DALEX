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
#' @param N number of observations used for calculations. By default all observations are taken.
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
predict_parts <- function(explainer, new_observation, N = NULL, ..., type = "break_down") {

  # Sample the data according to N

  switch (type,
          "break_down"              = predict_parts_break_down(explainer, new_observation, N = N, ...),
          "break_down_interactions" = predict_parts_break_down_interactions(explainer, new_observation, N = N, ...),
          "shap"                    = predict_parts_shap(explainer, new_observation, N = N, ...),
          "oscillations"            = predict_parts_oscillations(explainer, new_observation, N = N, ...),
          "oscillations_uni"        = predict_parts_oscillations_uni(explainer, new_observation, N = N, ...),
          "oscillations_emp"        = predict_parts_oscillations_emp(explainer, new_observation, N = N, ...),
          stop("The type argument shall be either 'shap' or 'break_down' or 'break_down_interactions' or 'oscillations' or 'oscillations_uni' or 'oscillations_emp'")
  )
}

#' @name predict_parts
#' @export
predict_parts_oscillations <- function(explainer, new_observation, N = NULL, ...) {
  # run checks against the explainer objects
  test_explainer(explainer, has_data = TRUE, function_name = "predict_parts_oscillations")

  # Cut data according to N
  explainer$data <- cut_data_to_n(explainer$data, N)

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
predict_parts_oscillations_uni <- function(explainer, new_observation, variable_splits_type = "uniform", N = NULL, ...) {
  # run checks against the explainer objects
  test_explainer(explainer, has_data = TRUE, function_name = "predict_parts_oscillations_uni")

  # Cut data according to N
  explainer$data <- cut_data_to_n(explainer$data, N)

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
predict_parts_oscillations_emp <- function(explainer, new_observation, variable_splits = NULL, variables = colnames(explainer$data), N = NULL, ...) {
  # run checks against the explainer objects
  test_explainer(explainer, has_data = TRUE, function_name = "predict_parts_oscillations_emp")
  variables <- intersect(variables, colnames(new_observation))

  # Cut data according to N
  explainer$data <- cut_data_to_n(explainer$data, N)

  variable_splits <- lapply(variables, function(var) {
    explainer$data[,var]
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
predict_parts_break_down <- function(explainer, new_observation, N = NULL, ...) {
  # run checks against the explainer objects
  test_explainer(explainer, has_data = TRUE, function_name = "predict_parts_break_down")

  # Cut data according to N
  explainer$data <- cut_data_to_n(explainer$data, N)

  # call the break_down
  res <- iBreakDown::break_down(explainer,
                                new_observation = new_observation,
                                ...)
  class(res) <- c('predict_parts', class(res))
  res
}

#' @name predict_parts
#' @export
predict_parts_break_down_interactions <- function(explainer, new_observation, N = NULL, ...) {
  # run checks against the explainer objects
  test_explainer(explainer, has_data = TRUE, function_name = "predict_parts_break_down_interactions")

  # Cut data according to N
  explainer$data <- cut_data_to_n(explainer$data, N)

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
predict_parts_shap <- function(explainer, new_observation, N = NULL, ...) {
  # run checks against the explainer objects
  test_explainer(explainer, has_data = TRUE, function_name = "predict_parts_shap")

  # Cut data according to N
  explainer$data <- cut_data_to_n(explainer$data, N)

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
