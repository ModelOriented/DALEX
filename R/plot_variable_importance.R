#' Plots Global Model Explanations (Variable Importance)
#'
#' Function \code{plot.variable_dropout_explainer} plots dropouts for variables used in the model.
#' It uses output from \code{variable_dropout} function that corresponds to permutation based measure of variable importance.
#' Variables are sorted in the same order in all panels. The order depends on the average drop out loss. In different panels variable contributions may not look like sorted if variable importance is different in different in different mdoels.
#'
#' @param x a variable dropout exlainer produced with the 'variable_dropout' function
#' @param ... other explainers that shall be plotted together
#' @param max_vars maximum number of variables that shall be presented for for each model
#' @param bar_width width of bars. By default 10
#' @param show_baseline logical. Should the baseline be included?
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
plot.variable_importance_explainer <- function(x, ..., max_vars = 10, bar_width = 10, show_baseline = FALSE, desc_sorting = TRUE) {

  if (!is.logical(desc_sorting)){
    stop("desc_sorting is not logical")
  }

  # combine all explainers in a single frame
  ## remove in 0.4  dfl <- c(list(x), list(...))
  ## remove in 0.4  expl_df <- do.call(rbind, dfl)
  expl_df <- combine_explainers(x, ...)

  # add an additional column that serve as a baseline
  bestFits <- expl_df[expl_df$variable == "_full_model_", ]
  ext_expl_df <- merge(expl_df, bestFits[,c("label", "dropout_loss")], by = "label")

  # set the order of variables depending on their contribution
  reorder_levels <- ext_expl_df$dropout_loss.x - ext_expl_df$dropout_loss.y

  ext_expl_df$variable <- reorder(ext_expl_df$variable,
                                  reorder_levels * ifelse(desc_sorting, 1, -1),
                                  mean)

  # for each model leave only max_vars
  trimmed_parts <- lapply(unique(ext_expl_df$label), function(label) {
    tmp <- ext_expl_df[ext_expl_df$label == label, ]
    tmp[tail(order(tmp$dropout_loss.x), max_vars), ]
  })
  ext_expl_df <- do.call(rbind, trimmed_parts)

  if (!show_baseline) {
    ext_expl_df <- ext_expl_df[ext_expl_df$variable != "_baseline_" &
                                 ext_expl_df$variable != "_full_model_", ]
  }

  variable <- dropout_loss.x <- dropout_loss.y <- dropout_loss <- label <- NULL
  nlabels <- length(unique(bestFits$label))
  # plot it
  ggplot(ext_expl_df, aes(variable, ymin = dropout_loss.y, ymax = dropout_loss.x, color = label)) +
    geom_hline(data = bestFits, aes(yintercept = dropout_loss, color = label), lty = 3) +
    geom_linerange(size = bar_width) + coord_flip() +
    scale_color_manual(values = colors_discrete_drwhy(nlabels)) +
    facet_wrap(~label, ncol = 1, scales = "free_y") + theme_drwhy_vertical() +
    theme(legend.position = "none") +
    ylab("Loss-drop after perturbations") + xlab("")
}

