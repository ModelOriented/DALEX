#' Preimplemented Loss Functions
#'
#' @param predicted predicted scores, either vector of matrix, these are returned from the model specific `predict_function()``
#' @param observed observed scores or labels, these are supplied as explainer specific `y`
#' @param p_min for cross entropy, minimal value for probability to make sure that `log` will not explode
#' @param na.rm logical, should missing values be removed?
#'
#' @return numeric - value of the loss function
#'
#' @aliases loss_cross_entropy loss_sum_of_squares loss_root_mean_square loss_accuracy
#' @export
#' @examples
#'  \dontrun{
#' library("randomForest")
#' HR_rf_model <- randomForest(as.factor(status == "fired")~., data = HR, ntree = 100)
#' loss_sum_of_squares(as.numeric(HR$status == "fired"), yhat(HR_rf_model))
#'  }
#' @export
loss_cross_entropy = function(observed, predicted, p_min = 0.0001, na.rm = TRUE) {
  p <- sapply(seq_along(observed), function(i)  predicted[i, observed[i]] )
  sum(-log(pmax(p, p_min)), na.rm = TRUE)
}

#' @export
loss_sum_of_squares = function(observed, predicted, na.rm = TRUE) sum((observed - predicted)^2, na.rm = na.rm)

#' @export
loss_root_mean_square = function(observed, predicted, na.rm = TRUE) sqrt(mean((observed - predicted)^2, na.rm = na.rm))

#' @export
loss_accuracy = function(observed, predicted, na.rm = TRUE) mean(observed == predicted, na.rm = na.rm)
