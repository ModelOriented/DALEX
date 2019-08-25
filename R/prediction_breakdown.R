#' Calculate Break Down Explanations
#'
#' This function is set deprecated. It is suggested to use \code{\link[iBreakDown]{break_down}} instead.
#' Find information how to use these functions here: \url{https://pbiecek.github.io/PM_VEE/breakDown.html}.
#'
#' @param explainer a model to be explained, preprocessed by the 'explain' function
#' @param observation a new observarvation for which predictions need to be explained
#' @param ... other parameters that will be passed to \code{breakDown::broken.default()}
#'
#' @return An object of the class 'single_prediction_explainer'.
#' It's a data frame with calculated average response.
#'
#' @aliases single_prediction
#' @references Predictive Models: Visual Exploration, Explanation and Debugging \url{https://pbiecek.github.io/PM_VEE/}
#' @export
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
#' dragon_lm_predict4 <- prediction_breakdown(dragon_lm_explainer4, observation = new_dragon)
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
#' dragon_rf_predict4 <- prediction_breakdown(dragon_rf_explainer4, observation = new_dragon)
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
#'  exp_sgn <- prediction_breakdown(explainer_gbm, observation = new_dragon)
#'  head(exp_sgn)
#'  plot(exp_sgn)
#'
#'  exp_sgn <- prediction_breakdown(explainer_gbm, observation = new_dragon, baseline = 0)
#'  plot(exp_sgn)
#'  }
#'
prediction_breakdown <- function(explainer, observation, ...) {
  # Deprecated, but print the message only once
  if (!exists("message_prediction_breakdown", envir = .DALEX.env)) {
    .DALEX.env$message_prediction_breakdown = TRUE
    .Deprecated("iBreakDown::break_down()", package = "iBreakDown", msg = "Please note that 'prediction_breakdown()' is now deprecated, it is better to use 'iBreakDown::break_down()' instead.\nFind examples and detailed introduction at: https://pbiecek.github.io/PM_VEE/breakDown.html")
  }


  if (!("explainer" %in% class(explainer))) stop("The prediction_breakdown() function requires an object created with explain() function.")
  if (is.null(explainer$data)) stop("The prediction_breakdown() function requires explainers created with specified 'data' parameter.")

  # breakDown
  res <- breakDown:::broken.default(explainer$model,
                new_observation = observation,
                data = explainer$data,
                predict.function = explainer$predict_function,
                ...)
  res$label <- rep(explainer$label, length(res$variable))

  class(res) <- c("prediction_breakdown_explainer", "data.frame")
  res

}

#' @export
single_prediction <- prediction_breakdown
