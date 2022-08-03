#' SHAP aggregated values
#'
#' This function works in a similar way to shap function from \code{iBreakDown} but it calculates explanations for a set of observation and then aggregates them.
#'
#' @param x an explainer created with function \code{\link[DALEX]{explain}} or a model.
#' @param data validation dataset, will be extracted from \code{x} if it is an explainer.
#' @param predict_function predict function, will be extracted from \code{x} if it is an explainer.
#' @param new_observations a set of new observations with columns that correspond to variables used in the model.
#' @param order if not \code{NULL}, then it will be a fixed order of variables. It can be a numeric vector or vector with names of variables.
#' @param ... other parameters.
#' @param label name of the model. By default it's extracted from the 'class' attribute of the model.
#'
#' @return an object of the \code{shap_aggregated} class.
#'
#' @references Explanatory Model Analysis. Explore, Explain and Examine Predictive Models. \url{https://ema.drwhy.ai}
#'
#'
#' @examples
#' library("DALEX")
#' set.seed(1313)
#' model_titanic_glm <- glm(survived ~ gender + age + fare,
#'                        data = titanic_imputed, family = "binomial")
#' explain_titanic_glm <- explain(model_titanic_glm,
#'                            data = titanic_imputed,
#'                            y = titanic_imputed$survived,
#'                            label = "glm")
#'
#' bd_glm <- shap_aggregated(explain_titanic_glm, titanic_imputed[1:10, ])
#' bd_glm
#' plot(bd_glm, max_features = 3)
#' @export
shap_aggregated <- function(explainer, new_observations, order = NULL, B = 25, ...) {    
  ret_raw <- data.frame(contribution = c(), variable_name = c(), label = c())
  
  for(i in 1:nrow(new_observations)){
    new_obs <- new_observations[i,]
    shap_vals <- iBreakDown::shap(explainer, new_observation = new_obs, B = B, ...)
    shap_vals <- shap_vals[shap_vals$B != 0, c('contribution', 'variable_name', 'label')]
    ret_raw <- rbind(ret_raw, shap_vals)
  }
  
  ret <- aggregate(ret_raw$contribution, list(ret_raw$variable_name, ret_raw$label), FUN=mean)
  colnames(ret) <- c('variable', 'label', 'contribution')
  rownames(ret) <- ret$variable   
  
  data_preds <- predict(explainer, explainer$data)
  mean_prediction <- mean(data_preds)
  
  if(is.null(order)) {
    ret <- ret[calculate_order(explainer, mean_prediction, new_observations, predict),]
  } else {
    ret <- ret[order,]
  }
  
  ret$position <- (nrow(ret) + 1):2
  ret$sign <- ifelse(ret$contribution >= 0, "1", "-1")
  
  ret <- rbind(ret, data.frame(variable = "intercept",
                               label = explainer$label,
                               contribution = mean_prediction,
                               position = max(ret$position) + 1,
                               sign = "X"), 
               make.row.names=FALSE)
  
  subset_preds <- predict(explainer, new_observations)
  mean_subset <- mean(subset_preds)
  ret <- rbind(ret, data.frame(variable = "prediction",
                               label = explainer$label,
                               contribution = mean_subset,
                               position = 1,
                               sign = "X"), 
               make.row.names=FALSE)
  
  ret <- ret[call_order_func(ret$position, decreasing = TRUE), ]
  
  ret$cumulative <- cumsum(ret$contribution)
  ret$cumulative[nrow(ret)] <- ret$contribution[nrow(ret)]
  
  ret$variable_name <- ret$variable
  ret$variable_name <- factor(ret$variable_name, levels=c(levels(ret$variable_name), ''))
  ret$variable_name[nrow(ret)] <- ''
  
  ret$variable_value <- '' # column for consistency    
  
  predictions_new <- data.frame(contribution = subset_preds, variable_name='prediction', label=ret$label[1])
  predictions_old <- data.frame(contribution = data_preds, variable_name='intercept', label=ret$label[1])
  ret_raw <- rbind(ret_raw, predictions_new, predictions_old)    
  
  out <- list(aggregated = ret, raw = ret_raw)
  class(out) <- c('shap_aggregated', class(out))
  
  out
}

call_order_func <- function(...) {
  order(...)
}

calculate_1d_changes <- function(model, new_observation, data, predict_function) {
  average_yhats <- list()
  j <- 1
  for (i in colnames(new_observation)) {
    current_data <- data
    current_data[,i] <- new_observation[,i]
    yhats <- predict_function(model, current_data)
    average_yhats[[j]] <- colMeans(as.data.frame(yhats))
    j <- j + 1
  }
  names(average_yhats) <- colnames(new_observation)
  average_yhats
}

generate_average_observation <- function(subset) {
  # takes average
  numeric_cols <- unlist(lapply(subset, is.numeric))
  numeric_cols <- names(numeric_cols[numeric_cols == TRUE])
  df_numeric <- t(as.data.frame(colMeans(subset[,numeric_cols])))
  
  # takes most frequent one
  factor_cols <- unlist(lapply(subset, is.factor))
  factor_cols <- names(factor_cols[factor_cols == TRUE])
  df_factory <- as.data.frame(lapply(factor_cols, function(col) {
    factor(names(which.max(table(subset[,col]))), levels = levels(subset[,col]))
  }))
  colnames(df_factory) <- factor_cols
  
  # also takes most frequent one
  other_cols <- unlist(lapply(subset, function(x){(!is.numeric(x)) & (!is.factor(x))}))
  other_cols <- names(other_cols[other_cols == TRUE])
  df_others <- as.data.frame(lapply(other_cols, function(col) {
    tab <- table(subset[,col])
    tab_names <- attr(tab, 'dimnames')[[1]]
    tab_names[which.max(tab)]
  }), stringsAsFactors = FALSE)
  colnames(df_others) <- other_cols
  
  outs <- list()
  if(!ncol(df_numeric) == 0){outs <- append(list(df_numeric), outs)}
  if(!ncol(df_factory) == 0){outs <- append(list(df_factory), outs)}
  if(!ncol(df_others) == 0){outs <- append(list(df_others), outs)}
  
  do.call("cbind", outs)[,colnames(subset)]
}

calculate_order <- function(x, mean_prediction, new_data, predict_function) {
  baseline_yhat <- mean_prediction
  
  generated_obs <- generate_average_observation(new_data)
  
  average_yhats <- calculate_1d_changes(x, generated_obs, x$data, predict_function)
  diffs_1d <- sapply(seq_along(average_yhats), function(i) {
    sqrt(mean((average_yhats[[i]] - baseline_yhat)^2))
  })
  
  order(diffs_1d, decreasing = TRUE)  
} 
