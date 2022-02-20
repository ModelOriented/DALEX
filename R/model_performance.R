#' Dataset Level Model Performance Measures
#'
#' Function \code{model_performance()} calculates various performance measures for classification and regression models.
#' For classification models following measures are calculated: F1, accuracy, recall, precision and AUC.
#' For regression models following measures are calculated: mean squared error, R squared, median absolute deviation.
#'
#' @param explainer a model to be explained, preprocessed by the \code{\link{explain}} function
#' @param ... other parameters
#' @param cutoff a cutoff for classification models, needed for measures like recall, precision, ACC, F1. By default 0.5.
#'
#' @return An object of the class \code{model_performance}.
#'
#' It's a list with following fields:
#'
#' \itemize{
#' \item \code{residuals} - data frame that contains residuals for each observation
#' \item \code{measures} - list with calculated measures that are dedicated for the task, whether it is regression, binary classification or multiclass classification.
#' \item \code{type} - character that specifies type of the task.
#' }
#'
#' @references Explanatory Model Analysis. Explore, Explain, and Examine Predictive Models. \url{https://ema.drwhy.ai/}
#' @importFrom stats median weighted.mean
#' @export
#' @examples
#' \donttest{
#' # regression
#'
#' library("ranger")
#' apartments_ranger_model <- ranger(m2.price~., data = apartments, num.trees = 50)
#' explainer_ranger_apartments  <- explain(apartments_ranger_model, data = apartments[,-1],
#'                              y = apartments$m2.price, label = "Ranger Apartments")
#' model_performance_ranger_aps <- model_performance(explainer_ranger_apartments )
#' model_performance_ranger_aps
#' plot(model_performance_ranger_aps)
#' plot(model_performance_ranger_aps, geom = "boxplot")
#' plot(model_performance_ranger_aps, geom = "histogram")
#'
#' # binary classification
#'
#' titanic_glm_model <- glm(survived~., data = titanic_imputed, family = "binomial")
#' explainer_glm_titanic <- explain(titanic_glm_model, data = titanic_imputed[,-8],
#'                          y = titanic_imputed$survived)
#' model_performance_glm_titanic <- model_performance(explainer_glm_titanic)
#' model_performance_glm_titanic
#' plot(model_performance_glm_titanic)
#' plot(model_performance_glm_titanic, geom = "boxplot")
#' plot(model_performance_glm_titanic, geom = "histogram")
#'
#' # multilabel classification
#'
#' HR_ranger_model <- ranger(status~., data = HR, num.trees = 50,
#'                                probability = TRUE)
#' explainer_ranger_HR  <- explain(HR_ranger_model, data = HR[,-6],
#'                              y = HR$status, label = "Ranger HR")
#' model_performance_ranger_HR <- model_performance(explainer_ranger_HR)
#' model_performance_ranger_HR
#' plot(model_performance_ranger_HR)
#' plot(model_performance_ranger_HR, geom = "boxplot")
#' plot(model_performance_ranger_HR, geom = "histogram")
#'
#'}
#'
model_performance <- function(explainer, ..., cutoff = 0.5) {
  test_explainer(explainer, has_data = TRUE, has_y = TRUE, function_name = "model_performance")

  # Check since explain could have been run with precalculate = FALSE
  if (is.null(explainer$y_hat)) {
    predicted <- explainer$predict_function(explainer$model, explainer$data, ...)
  } else {
    predicted <- explainer$y_hat
  }

  observed <- explainer$y
  # Check since explain could have been run with precalculate = FALSE
  if (is.null(explainer$residuals)) {
    diff <- explainer$residual_function(explainer$model, explainer$data, observed, explainer$predict_function)
  } else {
    # changed according to #130
    diff <- explainer$residuals
  }

  residuals <- data.frame(predicted, observed, diff = diff)

  # get proper measures
  type <- explainer$model_info$type
  if (type == "regression") {
    measures <- list(
      mse = model_performance_mse(predicted, observed),
      rmse = model_performance_rmse(predicted, observed),
      r2 = model_performance_r2(predicted, observed),
      mad = model_performance_mad(predicted, observed)
    )
  } else if (type == "classification") {
    tp = sum((observed == 1) * (predicted >= cutoff))
    fp = sum((observed == 0) * (predicted >= cutoff))
    tn = sum((observed == 0) * (predicted < cutoff))
    fn = sum((observed == 1) * (predicted < cutoff))

    measures <- list(
      recall            = model_performance_recall(tp, fp, tn, fn),
      precision         = model_performance_precision(tp, fp, tn, fn),
      f1                = model_performance_f1(tp, fp, tn, fn),
      accuracy          = model_performance_accuracy(tp, fp, tn, fn),
      auc               = model_performance_auc(predicted, observed)
      mcc               = model_performance_mcc(tp, fp, tn, fn),
      brier_score       = model_performance_brier(predicted, observed),
      log_loss          = model_performance_logloss(predicted, observed),
      balanced_accuracy = model_performance_bacc(tp, fp, tn, fn)
    )
  } else if (type == "multiclass") {
    measures <- list(
      micro_F1 = model_performance_micro_f1(predicted, observed),
      macro_F1 = model_performance_macro_f1(predicted, observed),
      w_macro_F1 = model_performance_weighted_macro_f1(predicted, observed),
      accuracy = model_performance_accuracy_multi(predicted, observed),
      w_macro_auc = model_performance_weighted_macro_auc(predicted, observed)
    )
  } else {
    stop("`explainer$model_info$type` should be one of ['regression', 'classification', 'multiclass'] - pass `model_info = list(type = $type$)` to the `explain` function. Submit an issue on https://github.com/ModelOriented/DALEX/issues if you think that this model should be covered by default.")
  }

  residuals$label <- explainer$label

  structure(list(residuals, measures, type),
            .Names = c("residuals", "measures", "type"),
            class = "model_performance")
}


model_performance_mse <- function(predicted, observed) {
  mean((predicted - observed)^2, na.rm = TRUE)
}

model_performance_rmse <- function(predicted, observed) {
  sqrt(mean((predicted - observed)^2, na.rm = TRUE))
}

model_performance_r2 <- function(predicted, observed) {
  1 - model_performance_mse(predicted, observed)/model_performance_mse(mean(observed), observed)
}

model_performance_mad <- function(predicted, observed) {
  median(abs(predicted - observed))
}

model_performance_auc <- function(predicted, observed) {
  tpr_tmp <- tapply(observed, predicted, sum)
  TPR <- c(0,cumsum(rev(tpr_tmp)))/sum(observed)
  fpr_tmp <- tapply(1 - observed, predicted, sum)
  FPR <- c(0,cumsum(rev(fpr_tmp)))/sum(1 - observed)

  auc <- sum(diff(FPR)*(TPR[-1] + TPR[-length(TPR)])/2)
  auc
}

model_performance_recall <- function(tp, fp, tn, fn) {
  tp/(tp + fn)
}

model_performance_precision <- function(tp, fp, tn, fn) {
  tp/(tp + fp)
}

model_performance_f1 <- function(tp, fp, tn, fn) {
  recall = tp/(tp + fn)
  precision = tp/(tp + fp)
  2 * (precision * recall)/(precision + recall)
}

model_performance_accuracy <- function(tp, fp, tn, fn) {
  (tp + tn)/(tp + fp + tn + fn)
}

model_performance_bacc <- function(tp, fp, tn, fn) {
  ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
}

model_performance_logloss <- function(predicted, observed) {
  -mean(observed * log(predicted) + (1 - observed) * log(1 - predicted)) 
}

model_performance_brier <- function(predicted, observed) {
  mean((predicted - observed) ^ 2)
}

model_performance_mcc <- function(tp, fp, tn, fn) {
  ((tp *  tn) - (fp * fn)) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
}


model_performance_macro_f1 <- function(predicted, observed) {
  predicted_vectorized <- turn_probs_into_vector(predicted)
  confusion_matrixes <- calculate_confusion_matrixes(predicted_vectorized, observed)
  f1_scores <- sapply(confusion_matrixes, function(x){
    model_performance_f1(x$tp, x$fp, x$tn, x$fn)
  })
  mean(f1_scores)
}

model_performance_micro_f1 <- function(predicted, observed) {
  # For case where each point can be assigned only to one class micro_f1 equals acc
  model_performance_accuracy_multi(predicted, observed)
}

model_performance_weighted_macro_f1 <- function(predicted, observed) {
  predicted_vectorized <- turn_probs_into_vector(predicted)
  confusion_matrixes <- calculate_confusion_matrixes(predicted_vectorized, observed)
  f1_scores <- sapply(confusion_matrixes, function(x){
    model_performance_f1(x$tp, x$fp, x$tn, x$fn)
  })
  weighted.mean(f1_scores, prop.table(table(observed))[names(confusion_matrixes)])
}

model_performance_accuracy_multi <- function(predicted, observed) {
  predicted_vectorized <- turn_probs_into_vector(predicted)
  mean(predicted_vectorized == observed)
}

model_performance_weighted_macro_auc <- function(predicted, observed) {
  observed <- as.character(observed)
  auc_scores <- sapply(unique(observed), function(x){
    model_performance_auc(predicted[,x], as.numeric(observed == x))
  })
  weighted.mean(auc_scores, prop.table(table(observed))[unique(observed)])
}

turn_probs_into_vector <- function(observed) {
  apply(observed, 1, function(x){
    colnames(observed)[which.max(x)]
  })
}

calculate_confusion_matrixes <- function(predicted, observed) {
  observed <- as.character(observed)
  ret <- lapply(unique(observed), function(x){
    tp <- mean(predicted[predicted == x] == observed[predicted == x])
    fp <- mean(predicted[predicted == x] != observed[predicted == x])
    tn <- mean(predicted[predicted != x] == observed[predicted != x])
    fn <- mean(predicted[predicted != x] != observed[predicted != x])
    list(tp = ifelse(is.nan(tp), 0, tp),
         fp = ifelse(is.nan(fp), 0, fp),
         tn = ifelse(is.nan(tn), 0, tn),
         fn = ifelse(is.nan(fn), 0, fn))
  })
  names(ret) <- unique(observed)
  ret
}
