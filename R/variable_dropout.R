#' Loss from Variable Dropout
#'
#' @param explainer a model to be explained, preprocessed by the 'explain' function
#' @param loss_function a function thet will be used to assess variable importance
#' @param ... other parameters
#' @param type character, type of transformation that should be applied for dropout loss. 'raw' results raw drop lossess, 'ratio' returns \code{drop_loss/drop_loss_full_model} while 'difference' returns \code{drop_loss - drop_loss_full_model}
#' @param n_sample number of observations that should be sampled for calculation of variable importance. If negative then variable importance will be calculated on whole dataset (no sampling).
#'
#' @return An object of the class 'variable_leverage_explainer'.
#' It's a data frame with calculated average response.
#'
#' @aliases variable_importance variable_dropout
#' @export
#' @examples
#'  \dontrun{
#' library("breakDown")
#' library("randomForest")
#' HR_rf_model <- randomForest(left~., data = breakDown::HR_data, ntree = 100)
#' explainer_rf  <- explain(HR_rf_model, data = HR_data, y = HR_data$left)
#' vd_rf <- variable_importance(explainer_rf, type = "raw")
#' vd_rf
#'
#' HR_glm_model <- glm(left~., data = breakDown::HR_data, family = "binomial")
#' explainer_glm <- explain(HR_glm_model, data = HR_data, y = HR_data$left)
#' logit <- function(x) exp(x)/(1+exp(x))
#' vd_glm <- variable_importance(explainer_glm, type = "raw",
#'                         loss_function = function(observed, predicted)
#'                                      sum((observed - logit(predicted))^2))
#' vd_glm
#'
#' library("xgboost")
#' model_martix_train <- model.matrix(left~.-1, breakDown::HR_data)
#' data_train <- xgb.DMatrix(model_martix_train, label = breakDown::HR_data$left)
#' param <- list(max_depth = 2, eta = 1, silent = 1, nthread = 2,
#'               objective = "binary:logistic", eval_metric = "auc")
#' HR_xgb_model <- xgb.train(param, data_train, nrounds = 50)
#' explainer_xgb <- explain(HR_xgb_model, data = model_martix_train,
#'                      y = HR_data$left, label = "xgboost")
#' vd_xgb <- variable_importance(explainer_xgb, type = "raw")
#' vd_xgb
#' plot(vd_xgb)
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

  variables <- colnames(explainer$data)
  if (n_sample > 0) {
    sampled_rows <- sample.int(nrow(explainer$data), n_sample, replace = TRUE)
  } else {
    sampled_rows <- 1:nrow(explainer$data)
  }
  sampled_data <- explainer$data[sampled_rows,]
  observed <- explainer$y[sampled_rows]

  loss_0 <- loss_function(observed,
                          explainer$predict_function(explainer$model, sampled_data))
  loss_full <- loss_function(sample(observed),
                          explainer$predict_function(explainer$model, sampled_data))
  res <- sapply(variables, function(variable) {
    ndf <- sampled_data
    ndf[,variable] <- sample(ndf[,variable])
    predicted <- explainer$predict_function(explainer$model, ndf)
    loss_function(observed, predicted)
  })
  res <- sort(res)
  res <- data.frame(variable = c("_full_model_",names(res), "_baseline_"),
                    dropout_loss = c(loss_0, res, loss_full))
  if (type == "ratio") {
    res$dropout_loss = res$dropout_loss / loss_0
  }
  if (type == "difference") {
    res$dropout_loss = res$dropout_loss - loss_0
  }

  class(res) <- c("variable_importance_explainer", "data.frame")
  res$label <- explainer$label
  res
}
#' @export
variable_dropout <- variable_importance

