#' Dataset Level Variable Importance as Change in Loss Function after Variable Permutations
#'
#' From DALEX version 1.0 this function calls the \code{\link[ingredients]{feature_importance}}
#' Find information how to use this function here: \url{http://ema.drwhy.ai/featureImportance.html}.
#'
#' @param explainer a model to be explained, preprocessed by the \code{explain} function
#' @param loss_function a function that will be used to assess variable importance. By default it is 1-AUC for classification, cross entropy for multilabel classification and RMSE for regression. Custom, user-made loss function should accept two obligatory parameters (observed, predicted), where \code{observed} states for actual values of the target, while \code{predicted} for predicted values. If attribute "loss_accuracy" is associated with function object, then it will be plotted as name of the loss function.
#' @param ... other parameters
#' @param type character, type of transformation that should be applied for dropout loss. \code{variable_importance} and \code{raw} results raw drop lossess, \code{ratio} returns \code{drop_loss/drop_loss_full_model} while \code{difference} returns \code{drop_loss - drop_loss_full_model}
#' @param N number of observations that should be sampled for calculation of variable importance. If \code{NULL} then variable importance will be calculated on whole dataset (no sampling).
#' @param n_sample alias for \code{N} held for backwards compatibility. number of observations that should be sampled for calculation of variable importance.
#'
#' @references Explanatory Model Analysis. Explore, Explain and Examine Predictive Models. \url{http://ema.drwhy.ai/}
#' @return An object of the class \code{feature_importance}.
#' It's a data frame with calculated average response.
#'
#' @aliases variable_importance feature_importance model_parts
#'
#' @import ggplot2
#' @importFrom stats model.frame reorder
#' @export
#'
#' @examples
#' \donttest{
#' # regression
#'
#' library("ranger")
#' apartments_ranger_model <- ranger(m2.price~., data = apartments, num.trees = 50)
#' explainer_ranger  <- explain(apartments_ranger_model, data = apartments[,-1],
#'                              y = apartments$m2.price, label = "Ranger Apartments")
#' model_parts_ranger_aps <- model_parts(explainer_ranger, type = "raw")
#' head(model_parts_ranger_aps, 8)
#' plot(model_parts_ranger_aps)
#'
#' # binary classification
#'
#' titanic_glm_model <- glm(survived~., data = titanic_imputed, family = "binomial")
#' explainer_glm_titanic <- explain(titanic_glm_model, data = titanic_imputed[,-8],
#'                          y = titanic_imputed$survived)
#' logit <- function(x) exp(x)/(1+exp(x))
#' custom_loss <- function(observed, predicted){
#'    sum((observed - logit(predicted))^2)
#' }
#' attr(custom_loss, "loss_name") <- "Logit residuals"
#' model_parts_glm_titanic <- model_parts(explainer_glm_titanic, type = "raw",
#'                                        loss_function = custom_loss)
#' head(model_parts_glm_titanic, 8)
#' plot(model_parts_glm_titanic)
#'
#' # multilabel classification
#'
#' HR_ranger_model_HR <- ranger(status~., data = HR, num.trees = 50,
#'                                probability = TRUE)
#' explainer_ranger_HR  <- explain(HR_ranger_model_HR, data = HR[,-6],
#'                              y = HR$status, label = "Ranger HR")
#' model_parts_ranger_HR <- model_parts(explainer_ranger_HR, type = "raw")
#' head(model_parts_ranger_HR, 8)
#' plot(model_parts_ranger_HR)
#'
#'}
#'
model_parts <- function(explainer,
                              loss_function = loss_default(explainer$model_info$type),
                              ...,
                              type = "variable_importance",
                              N = n_sample,
                              n_sample = 1000) {
  # run checks against the explainer objects
  test_explainer(explainer, has_data = TRUE, has_y = TRUE, function_name = "model_parts")
  if (!(type %in% c("difference", "ratio", "raw", "variable_importance"))) stop("Type shall be one of 'variable_importance', 'difference', 'ratio', 'raw'")
  if (type == "variable_importance") type <- "raw" #it's an alias

  res <- ingredients::feature_importance(x = explainer,
                                         loss_function = loss_function,
                                         type = type,
                                         N = N,
                                         ...)
  class(res) <- c('model_parts', class(res))
  res
}
#' @export
feature_importance <- model_parts

#' @export
variable_importance <- model_parts

