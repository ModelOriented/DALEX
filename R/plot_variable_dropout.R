#' Plots Global Model Explanations (Variable Drop-out)
#'
#' Function \code{plot.variable_dropout_explainer} plots dropouts for variables used in the model.
#' It uses output from \code{variable_dropout} function that corresponds to permutation based measure of variable importance.
#' Variables are sorted in the same order in all panels. The order depends on the average drop out loss. In different panels variable contributions may not look like sorted if variable importance is different in different in different mdoels.
#'
#' @param x a variable dropout exlainer produced with the 'variable_dropout' function
#' @param ... other explainers that shall be plotted together
#' @param max_vars maximum number of variables that shall be presented for for each model
#'
#' @importFrom stats model.frame reorder
#' @return a ggplot2 object
#' @export
#'
#' @examples
#'
#' #\dontrun{
#' library("breakDown")
#' library("randomForest")
#' HR_rf_model <- randomForest(left~., data = breakDown::HR_data, ntree = 100)
#' explainer_rf  <- explain(HR_rf_model, data = HR_data, y = HR_data$left)
#' vd_rf <- variable_dropout(explainer_rf, type = "raw")
#' vd_rf
#' plot(vd_rf)
#'
#' HR_glm_model <- glm(left~., data = breakDown::HR_data, family = "binomial")
#' explainer_glm <- explain(HR_glm_model, data = HR_data, y = HR_data$left)
#' logit <- function(x) exp(x)/(1+exp(x))
#' vd_glm <- variable_dropout(explainer_glm, type = "raw",
#'                         loss_function = function(observed, predicted)
#'                                    sum((observed - logit(predicted))^2))
#' vd_glm
#' plot(vd_glm)
#'
#' library("xgboost")
#' model_martix_train <- model.matrix(left~.-1, breakDown::HR_data)
#' data_train <- xgb.DMatrix(model_martix_train, label = breakDown::HR_data$left)
#' param <- list(max_depth = 2, eta = 1, silent = 1, nthread = 2,
#'               objective = "binary:logistic", eval_metric = "auc")
#' HR_xgb_model <- xgb.train(param, data_train, nrounds = 50)
#' explainer_xgb <- explain(HR_xgb_model, data = model_martix_train,
#'                                     y = HR_data$left, label = "xgboost")
#' vd_xgb <- variable_dropout(explainer_xgb, type = "raw")
#' vd_xgb
#' plot(vd_xgb)
#'
#' plot(vd_rf, vd_glm, vd_xgb)
#'
#' # NOTE:
#' # if you like to have all importances hooked to 0, you can do this as well
#' vd_rf <- variable_dropout(explainer_rf, type = "difference")
#' vd_glm <- variable_dropout(explainer_glm, type = "difference",
#'                         loss_function = function(observed, predicted)
#'                                    sum((observed - logit(predicted))^2))
#' vd_xgb <- variable_dropout(explainer_xgb, type = "difference")
#' plot(vd_rf, vd_glm, vd_xgb)
#' #}
#'
plot.variable_dropout_explainer <- function(x, ..., max_vars = 10) {
  dfl <- c(list(x), list(...))

  # combine all explainers in a single frame
  expl_df <- do.call(rbind, dfl)

  # add an additional column that serve as a baseline
  bestFits <- expl_df[expl_df$variable == "_full_model_", ]
  ext_expl_df <- merge(expl_df, bestFits[,c("label", "dropout_loss")], by = "label")

  # set the order of variables depending on their contribution
  ext_expl_df$variable <- reorder(ext_expl_df$variable,
                                  ext_expl_df$dropout_loss.x - ext_expl_df$dropout_loss.y,
                                  mean)

  # for each model leave only max_vars
  trimmed_parts <- lapply(unique(ext_expl_df$label), function(label) {
    tmp <- ext_expl_df[ext_expl_df$label == label, ]
    tmp[tail(order(tmp$dropout_loss.x), max_vars), ]
  })
  ext_expl_df <- do.call(rbind, trimmed_parts)

  variable <- dropout_loss.x <- dropout_loss.y <- NULL

  # plot it
  ggplot(ext_expl_df, aes(variable, ymin = dropout_loss.y, ymax = dropout_loss.x)) +
    geom_errorbar() + coord_flip() +
    facet_wrap(~label, ncol = 1, scales = "free_y") +
    ylab("Drop-out loss") + xlab("") +
    theme_mi2()

}

