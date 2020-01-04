#' Plot Variable Importance Explanations
#'
#' Function \code{plot.variable_importance_explainer} plots permutational importance
#' of variables used in the model.
#' It uses output from \code{variable_importance} function that corresponds
#' to permutation based measure of variable importance.
#' Variables are sorted in the same order in all panels.
#' In different panels variable contributions may not look like sorted
#' if variable importance is different in different models.
#'
#' @param x a variable importance explainer produced with the \code{\link{variable_importance}} function
#' @param ... other explainers that shall be plotted together
#' @param max_vars maximum number of variables that shall be presented for for each model
#' @param bar_width width of bars. By default 10
#' @param show_boxplots logical if TRUE (default) boxplot will be plotted to show permutation data.
#' @param desc_sorting logical. Should the bars be sorted descending? By default TRUE
#'
#' @importFrom stats model.frame reorder
#' @return a ggplot2 object
#' @export
#'
#' @examples
#'
#'  \dontrun{
#' library("randomForest")
#' HR_rf_model <- randomForest(as.factor(status == "fired")~., data = HR, ntree = 100)
#' explainer_rf  <- explain(HR_rf_model, data = HR, y = HR$status == "fired")
#' vd_rf <- variable_importance(explainer_rf, type = "raw")
#' head(vd_rf)
#' plot(vd_rf)
#'
#' HR_glm_model <- glm(status == "fired"~., data = HR, family = "binomial")
#' explainer_glm <- explain(HR_glm_model, data = HR, y = HR$status == "fired")
#' logit <- function(x) exp(x)/(1+exp(x))
#' vd_glm <- variable_importance(explainer_glm, type = "raw",
#'                         loss_function = function(observed, predicted)
#'                                    sum((observed - logit(predicted))^2))
#' head(vd_glm)
#' plot(vd_glm)
#'
#' library("xgboost")
#' model_martix_train <- model.matrix(status == "fired"~.-1, HR)
#' data_train <- xgb.DMatrix(model_martix_train, label = HR$status == "fired")
#' param <- list(max_depth = 2, eta = 1, silent = 1, nthread = 2,
#'               objective = "binary:logistic", eval_metric = "auc")
#' HR_xgb_model <- xgb.train(param, data_train, nrounds = 50)
#' explainer_xgb <- explain(HR_xgb_model, data = model_martix_train,
#'                                     y = HR$status == "fired", label = "xgboost")
#' vd_xgb <- variable_importance(explainer_xgb, type = "raw")
#' head(vd_xgb)
#' plot(vd_xgb)
#'
#' plot(vd_rf, vd_glm, vd_xgb, bar_width = 4)
#'
#' # NOTE:
#' # if you like to have all importances hooked to 0, you can do this as well
#' vd_rf <- variable_importance(explainer_rf, type = "difference")
#' vd_glm <- variable_importance(explainer_glm, type = "difference",
#'                         loss_function = function(observed, predicted)
#'                                    sum((observed - logit(predicted))^2))
#' vd_xgb <- variable_importance(explainer_xgb, type = "difference")
#' plot(vd_rf, vd_glm, vd_xgb, bar_width = 4)
#'  }
#'
plot.variable_importance_explainer <- function(x, ..., max_vars = 10, show_boxplots = TRUE, bar_width = 10, desc_sorting = TRUE) {
  # from DALEX version 0.5 this function is a copy of ingredients::plot.feature_importance_explainer
  ingredients:::plot.feature_importance_explainer(x, ...,
                       max_vars = max_vars, show_boxplots = show_boxplots,
                       bar_width = bar_width,
                       desc_sorting = desc_sorting)
}


