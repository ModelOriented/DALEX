#' Dataset Level Variable Importance as Change in Loss Function after Variable Permutations
#'
#' From DALEX version 1.0 this function calls the \code{\link[ingredients]{feature_importance}}
#' Find information how to use this function here: \url{https://pbiecek.github.io/ema/featureImportance.html}.
#'
#' @param explainer a model to be explained, preprocessed by the 'explain' function
#' @param loss_function a function that will be used to assess variable importance
#' @param ... other parameters
#' @param type character, type of transformation that should be applied for dropout loss. 'raw' results raw drop lossess, 'ratio' returns \code{drop_loss/drop_loss_full_model} while 'difference' returns \code{drop_loss - drop_loss_full_model}
#' @param n_sample number of observations that should be sampled for calculation of variable importance. If negative then variable importance will be calculated on whole dataset (no sampling).
#'
#' @references Explanatory Model Analysis. Explore, Explain and Examine Predictive Models. \url{https://pbiecek.github.io/ema/}
#' @return An object of the class 'feature_importance'.
#' It's a data frame with calculated average response.
#'
#' @aliases variable_importance feature_importance
#' @import ggplot2
#' @importFrom stats model.frame reorder
#' @export
#' @examples
#'  \dontrun{
#' library("ranger")
#' titanic_ranger_model <- ranger(survived~., data = titanic_imputed, num.trees = 50,
#'                                probability = TRUE)
#' explainer_ranger  <- explain(titanic_ranger_model, data = titanic_imputed[,-8],
#'                              y = titanic_imputed$survived)
#' vi_ranger <- variable_importance(explainer_ranger, type = "raw")
#' head(vi_ranger, 8)
#' plot(vi_ranger)
#'
#' titanic_glm_model <- glm(survived~., data = titanic_imputed, family = "binomial")
#' explainer_glm <- explain(titanic_glm_model, data = titanic_imputed[,-8],
#'                          y = titanic_imputed$survived)
#' logit <- function(x) exp(x)/(1+exp(x))
#' vi_glm <- variable_importance(explainer_glm, type = "raw",
#'                         loss_function = function(observed, predicted)
#'                                      sum((observed - logit(predicted))^2))
#' head(vi_glm, 8)
#' plot(vi_glm)
#'  }
#'
variable_importance <- function(explainer,
                              loss_function = loss_sum_of_squares,
                              ...,
                              type = "raw",
                              n_sample = 1000) {
  # run checks against the explainer objects
  test_expaliner(explainer, has_data = TRUE, has_y = TRUE, function_name = "variable_importance")
  if (!(type %in% c("difference", "ratio", "raw"))) stop("Type shall be one of 'difference', 'ratio', 'raw'")

  ingredients::feature_importance(x = explainer,
                                  loss_function = loss_function,
                                  type = type,
                                  n_sample = n_sample,
                                  ...)
}
#' @export
feature_importance <- variable_importance

