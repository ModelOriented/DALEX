#' Calculate Variable Importance Explanations as Change in Loss Function after Variable Permutations
#'
#' From DALEX version 0.5 this function calls the \code{\link[ingredients]{feature_importance}}
#' Find information how to use this function here: \url{https://pbiecek.github.io/PM_VEE/featureImportance.html}.
#'
#' @param explainer a model to be explained, preprocessed by the 'explain' function
#' @param loss_function a function thet will be used to assess variable importance
#' @param ... other parameters
#' @param type character, type of transformation that should be applied for dropout loss. 'raw' results raw drop lossess, 'ratio' returns \code{drop_loss/drop_loss_full_model} while 'difference' returns \code{drop_loss - drop_loss_full_model}
#' @param n_sample number of observations that should be sampled for calculation of variable importance. If negative then variable importance will be calculated on whole dataset (no sampling).
#'
#' @references Predictive Models: Visual Exploration, Explanation and Debugging \url{https://pbiecek.github.io/PM_VEE/}
#' @return An object of the class 'feature_importance'.
#' It's a data frame with calculated average response.
#'
#' @aliases variable_importance feature_importance
#' @export
#' @examples
#'  \dontrun{
#' library(DALEX)
#' library("randomForest")
#' HR_rf_model <- randomForest(as.factor(status == "fired")~., data = HR, ntree = 100)
#' explainer_rf  <- explain(HR_rf_model, data = HR, y = HR$status == "fired")
#' vd_rf <- variable_importance(explainer_rf, type = "raw")
#' head(vd_rf, 8)
#'
#' HR_glm_model <- glm(as.factor(status == "fired")~., data = HR, family = "binomial")
#' explainer_glm <- explain(HR_glm_model, data = HR, y = HR$status == "fired")
#' logit <- function(x) exp(x)/(1+exp(x))
#' vd_glm <- variable_importance(explainer_glm, type = "raw",
#'                         loss_function = function(observed, predicted)
#'                                      sum((observed - logit(predicted))^2))
#' head(vd_glm, 8)
#'
#' library("xgboost")
#' model_martix_train <- model.matrix(status == "fired" ~ .-1, HR)
#' data_train <- xgb.DMatrix(model_martix_train, label = HR$status == "fired")
#' param <- list(max_depth = 2, eta = 1, silent = 1, nthread = 2,
#'               objective = "binary:logistic", eval_metric = "auc")
#' HR_xgb_model <- xgb.train(param, data_train, nrounds = 50)
#' explainer_xgb <- explain(HR_xgb_model, data = model_martix_train,
#'                      y = HR$status == "fired", label = "xgboost")
#' vd_xgb <- variable_importance(explainer_xgb, type = "raw")
#' head(vd_xgb, 8)
#'  }
#'
variable_importance <- function(explainer,
                              loss_function = loss_sum_of_squares,
                              ...,
                              type = "raw",
                              n_sample = 1000) {
  if (!("explainer" %in% class(explainer))) stop("The variable_importance() function requires an object created with explain() function.")
  if (is.null(explainer$data)) stop("The variable_importance() function requires explainers created with specified 'data' parameter.")
  if (is.null(explainer$y)) stop("The variable_importance() function requires explainers created with specified 'y' parameter.")
  if (!(type %in% c("difference", "ratio", "raw"))) stop("Type shall be one of 'difference', 'ratio', 'raw'")

  ingredients::feature_importance(x = explainer,
                                  loss_function = loss_function,
                                  type = type,
                                  n_sample = n_sample,
                                  ...)
}
#' @export
feature_importance <- variable_importance

