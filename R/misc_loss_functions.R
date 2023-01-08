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
  mean(observed == predicted, na.rm = na.rm) # this works for classes not probabilities
attr(loss_accuracy, "loss_name") <- "Accuracy"

#' @rdname loss_functions
#' @export
loss_one_minus_accuracy <-  function(observed, predicted, cutoff = 0.5, na.rm = TRUE) {
  tp = sum((observed == 1) * (predicted >= cutoff), na.rm = na.rm)
  fp = sum((observed == 0) * (predicted >= cutoff), na.rm = na.rm)
  tn = sum((observed == 0) * (predicted < cutoff), na.rm = na.rm)
  fn = sum((observed == 1) * (predicted < cutoff), na.rm = na.rm)

  acc <- (tp + tn)/(tp + fp + tn + fn)

  1 - acc
}
attr(loss_one_minus_accuracy, "loss_name") <- "One minus Accuracy"

#' @rdname loss_functions
#' @export
get_loss_one_minus_accuracy <- function(cutoff = 0.5, na.rm = TRUE) {
  function(o, p) loss_one_minus_accuracy(o, p, cutoff = cutoff, na.rm = na.rm)
}

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
get_loss_default <- function(x) {
  # explainer is an explainer or type of an explainer
  if ("explainer" %in% class(x))  x <- x$model_info$type
  switch (x,
          "regression"     = loss_root_mean_square,
          "classification" = loss_one_minus_auc,
          "multiclass"     = loss_cross_entropy,
          stop("`explainer$model_info$type` should be one of ['regression', 'classification', 'multiclass'] - pass `model_info = list(type = $type$)` to the `explain` function. Submit an issue on https://github.com/ModelOriented/DALEX/issues if you think that this model should be covered by default.")
  )
}

#' @export
loss_default <- function(x) {
  warn("`loss_default()` is deprecated; use `get_loss_default()` instead.")
  get_loss_default(x)
}



#' Wrapper for Loss Functions from the yardstick Package
#'
#' The yardstick package provides many auxiliary functions for calculating
#' the predictive performance of the model. However, they have an interface
#' that is consistent with the tidyverse philosophy. The loss_yardstick
#' function adapts loss functions from the yardstick package to functions
#' understood by DALEX. Type compatibility for y-values and for predictions
#' must be guaranteed by the user.
#'
#' @param loss loss function from the \code{yardstick} package
#' @param reverse shall the metric be reversed? for loss metrics lower values are better. \code{reverse = TRUE} is useful for accuracy-like metrics
#' @param reference if the metric is reverse then it is calculated as \code{reference - loss}. The default value is 1.
#'
#' @return loss function that can be used in the model_parts function
#'
#' @export
#' @examples
#'  \donttest{
#'  titanic_glm_model <- glm(survived~., data = titanic_imputed, family = "binomial")
#'  explainer_glm <- DALEX::explain(titanic_glm_model,
#'                                  data = titanic_imputed[,-8],
#'                                  y = factor(titanic_imputed$survived))
#'  # See the 'How to use DALEX with the yardstick package' vignette
#'  # which explains this model with measures implemented in the 'yardstick' package
#' }
#'
#' @rdname get_loss_yardstick
#' @export
get_loss_yardstick <- function(loss, reverse = FALSE, reference = 1) {
  # wrapper for yardstick loss functions
  if (reverse) {
    custom_loss <- function(observed, predicted) {
      df <- data.frame(observed, predicted)
      reference - loss(df, observed, predicted)$.estimate
    }
    attr(custom_loss, "loss_name") <- paste0(reference, " - ", deparse(substitute(loss)))
  } else {
    custom_loss <- function(observed, predicted) {
      df <- data.frame(observed, predicted)
      colnames(df) <- c("observed", "predicted")
      loss(df, observed, predicted)$.estimate
    }
    attr(custom_loss, "loss_name") <- deparse(substitute(loss))
  }

  custom_loss
}

#' @export
loss_yardstick <- function(loss, reverse = FALSE, reference = 1) {
  warn("`loss_yardstick()` is deprecated; use `get_loss_yardstick()` instead.")
  get_loss_yardstick(loss = loss, reverse = referese, reference = reference)
}

