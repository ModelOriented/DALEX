#' Calculate Loss Functions
#'
#' @param predicted predicted scores, either vector of matrix, these are returned from the model specific \code{predict_function()}
#' @param observed observed scores or labels, these are supplied as explainer specific \code{y}
#' @param p_min for cross entropy, minimal value for probability to make sure that \code{log} will not explode
#' @param na.rm logical, should missing values be removed?
#' @param x either an explainer or type of the model. One of "regression", "classification", "multiclass".
#'
#' @return numeric - value of the loss function
#'
#' @aliases loss_cross_entropy loss_sum_of_squares loss_root_mean_square loss_accuracy loss_one_minus_auc
#' @export
#' @examples
#'  \donttest{
#' library("ranger")
#' titanic_ranger_model <- ranger(survived~., data = titanic_imputed, num.trees = 50,
#'                                probability = TRUE)
#' loss_one_minus_auc(titanic_imputed$survived, yhat(titanic_ranger_model, titanic_imputed))
#'
#' HR_ranger_model_multi <- ranger(status~., data = HR, num.trees = 50, probability = TRUE)
#' loss_cross_entropy(as.numeric(HR$status), yhat(HR_ranger_model_multi, HR))
#'
#'  }
#' @rdname loss_functions
#' @export
loss_cross_entropy <- function(observed, predicted, p_min = 0.0001, na.rm = TRUE) {
  p <- sapply(seq_along(observed), function(i)  predicted[i, observed[i]] )
  sum(-log(pmax(p, p_min)), na.rm = TRUE)
}
attr(loss_cross_entropy, "loss_name") <- "Cross entropy"




#' @rdname loss_functions
#' @export
loss_sum_of_squares <- function(observed, predicted, na.rm = TRUE)
  sum((observed - predicted)^2, na.rm = na.rm)
attr(loss_sum_of_squares, "loss_name") <- "Sum of squared residuals (SSR)"

#' @rdname loss_functions
#' @export
loss_root_mean_square <- function(observed, predicted, na.rm = TRUE)
  sqrt(mean((observed - predicted)^2, na.rm = na.rm))
attr(loss_root_mean_square, "loss_name") <- "Root mean square error (RMSE)"

#' @rdname loss_functions
#' @export
loss_accuracy <-  function(observed, predicted, na.rm = TRUE)
  mean(observed == predicted, na.rm = na.rm)
attr(loss_accuracy, "loss_name") <- "Accuracy"

#' @rdname loss_functions
#' @export
loss_one_minus_auc <- function(observed, predicted){
  tpr_tmp <- tapply(observed, predicted, sum)
  TPR <- c(0,cumsum(rev(tpr_tmp)))/sum(observed)
  fpr_tmp <- tapply(1 - observed, predicted, sum)
  FPR <- c(0,cumsum(rev(fpr_tmp)))/sum(1 - observed)

  auc <- sum(diff(FPR)*(TPR[-1] + TPR[-length(TPR)])/2)

  1 - auc
}
attr(loss_one_minus_auc, "loss_name") <- "One minus AUC"

#' @rdname loss_functions
#' @export
loss_default <- function(x) {
  # explainer is an explainer or type of an explainer
  if ("explainer" %in% class(x))  x <- x$model_info$type
  switch (x,
          "regression"     = loss_root_mean_square,
          "classification" = loss_one_minus_auc,
          "multiclass"     = loss_cross_entropy,
          stop("`explainer$model_info$type` should be one of ['regression', 'classification', 'multiclass'] - pass `model_info = list(type = $type$)` to the `explain` function. Submit an issue on https://github.com/ModelOriented/DALEX/issues if you think that this model should be covered by default.")
  )
}




#' Wrapper for Loss Functions from the Yarstick Package
#'
#' The yardstick package provides many auxiliary functions for calculating
#' the predictive performance of the model. However, they have an interface
#' that is consistent with the tidyverse philosophy. The loss_yardstick
#' function adapts loss functions from the yardstick package to functions
#' understood by DALEX. Type compatibility for y-values and for predictions
#' must be guaranteed by the user.
#'
#' @param loss loss function from the yardstick package
#'
#' @return loss function that can be used in the model_parts function
#'
#' @export
#' @examples
#'  \donttest{
#'  # Classification Metrics
#'  # y and y_hat are factors!!!
#'  library("yardstick")
#'
#'  titanic_glm_model <- glm(survived~., data = titanic_imputed, family = "binomial")
#'  explainer_glm <- DALEX::explain(titanic_glm_model,
#'                                  data = titanic_imputed[,-8],
#'                                  y = factor(titanic_imputed$survived),
#'                                  predict_function = function(m, x) {
#'                                    factor((predict(m, x, type = "response") > 0.5) + 0)
#'                                  })
#'
#' model_parts_accuracy <- model_parts(explainer_glm, type = "raw",
#'                              loss_function = loss_yardstick(accuracy))
#' plot(model_parts_accuracy)
#'
#' # Class Probability Metrics
#' # y is a factor while y_hat is a numeric!!!
#'
#'  titanic_glm_model <- glm(survived~., data = titanic_imputed, family = "binomial")
#'  explainer_glm <- DALEX::explain(titanic_glm_model,
#'                                  data = titanic_imputed[,-8],
#'                                  y = factor(titanic_imputed$survived))
#'
#' model_parts_accuracy <- model_parts(explainer_glm, type = "raw",
#'                              loss_function = loss_yardstick(roc_auc))
#' plot(model_parts_accuracy)
#'
#' # Regression Metrics
#' # y and y_hat are numeric!!!
#'
#' library("ranger")
#' apartments_ranger <- ranger(m2.price~., data = apartments, num.trees = 50)
#' explainer_ranger  <- DALEX::explain(apartments_ranger, data = apartments[,-1],
#'                                     y = apartments$m2.price, label = "Ranger Apartments")
#' model_parts_ranger <- model_parts(explainer_ranger, type = "raw",
#'                        loss_function = loss_yardstick(rsq))
#' plot(model_parts_ranger)
#'
#' }
#'
#' @rdname loss_yardstick
#' @export
loss_yardstick <- function(loss) {
  # wrapper for yardstick loss functions
  custom_loss <- function(observed, predicted) {
    df <- data.frame(observed, predicted)
    loss(df, observed, predicted)$.estimate
  }
  attr(custom_loss, "loss_name") <- deparse(substitute(loss))
  custom_loss
}


