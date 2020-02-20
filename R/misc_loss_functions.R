#' Calculate Loss Functions
#'
#' @param predicted predicted scores, either vector of matrix, these are returned from the model specific \code{predict_function()}
#' @param observed observed scores or labels, these are supplied as explainer specific \code{y}
#' @param p_min for cross entropy, minimal value for probability to make sure that \code{log} will not explode
#' @param na.rm logical, should missing values be removed?
#'
#' @return numeric - value of the loss function
#'
#' @aliases loss_cross_entropy loss_sum_of_squares loss_root_mean_square loss_accuracy loss_one_minus_auc
#' @export
#' @examples
#'  \dontrun{
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

#' @rdname loss_functions
#' @export
loss_sum_of_squares <- function(observed, predicted, na.rm = TRUE)
  sum((observed - predicted)^2, na.rm = na.rm)

#' @rdname loss_functions
#' @export
loss_root_mean_square <- function(observed, predicted, na.rm = TRUE)
  sqrt(mean((observed - predicted)^2, na.rm = na.rm))

#' @rdname loss_functions
#' @export
loss_accuracy <-  function(observed, predicted, na.rm = TRUE)
  mean(observed == predicted, na.rm = na.rm)

#' @rdname loss_functions
#' @export
# Alicja Gosiewska (agosiewska) is the author of this function
loss_one_minus_auc <- function(observed, predicted){

  pred <- data.frame(fitted.values = predicted,
             y = observed)
  pred_sorted <- pred[order(pred$fitted.values, decreasing = TRUE), ]
  roc_y <- factor(pred_sorted$y)
  levels <- levels(roc_y)
  x <- cumsum(roc_y == levels[1])/sum(roc_y == levels[1])
  y <- cumsum(roc_y == levels[2])/sum(roc_y == levels[2])
  auc <- sum((x[2:length(roc_y)]  -x[1:length(roc_y)-1]) * y[2:length(roc_y)])
  1 - auc

}


