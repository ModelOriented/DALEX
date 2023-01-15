#' SHAP aggregated values
#'
#' This function works in a similar way to shap function from \code{iBreakDown} but it calculates explanations for a set of observation and then aggregates them.
#'
#' @param explainer a model to be explained, preprocessed by the \code{explain} function
#' @param new_observations a set of new observations with columns that correspond to variables used in the model.
#' @param order if not \code{NULL}, then it will be a fixed order of variables. It can be a numeric vector or vector with names of variables.
#' @param ... other parameters like \code{label}, \code{predict_function}, \code{data}, \code{x}
#' @param B number of random paths
#'
#' @return an object of the \code{shap_aggregated} class.
#'
#' @references Explanatory Model Analysis. Explore, Explain and Examine Predictive Models. \url{https://ema.drwhy.ai}
#' @importFrom stats aggregate
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

  data_preds <- predict(explainer, explainer$data)
  mean_prediction <- mean(data_preds)

  subset_preds <- predict(explainer, new_observations)
  mean_subset <- mean(subset_preds)

  if(is.null(order)) {
    order <- calculate_order(explainer, mean_prediction, new_observations, predict)
  }

  ret <- raw_to_aggregated(ret_raw, mean_prediction, mean_subset, order, explainer$label)

  predictions_new <- data.frame(contribution = subset_preds, variable_name='prediction', label=ret$label[1])
  predictions_old <- data.frame(contribution = data_preds, variable_name='intercept', label=ret$label[1])
  ret_raw <- rbind(ret_raw, predictions_new, predictions_old)

  out <- list(aggregated = ret, raw = ret_raw)
  class(out) <- c('shap_aggregated', class(out))

  out
}

raw_to_aggregated <- function(ret_raw, mean_prediction, mean_subset, order, label){
  ret <- aggregate(ret_raw$contribution, list(ret_raw$variable_name, ret_raw$label), FUN=mean)
  colnames(ret) <- c('variable', 'label', 'contribution')
  ret$variable <- as.character(ret$variable)
  rownames(ret) <- ret$variable

  ret <- ret[order,]

  ret$position <- (nrow(ret) + 1):2
  ret$sign <- ifelse(ret$contribution >= 0, "1", "-1")

  ret <- rbind(ret, data.frame(variable = "intercept",
                               label = label,
                               contribution = mean_prediction,
                               position = max(ret$position) + 1,
                               sign = "X"),
               make.row.names=FALSE)

  ret <- rbind(ret, data.frame(variable = "prediction",
                               label = label,
                               contribution = mean_subset,
                               position = 1,
                               sign = "X"),
               make.row.names=FALSE)

  ret <- ret[call_order_func(ret$position, decreasing = TRUE), ]

  ret$cumulative <- cumsum(ret$contribution)
  ret$cumulative[nrow(ret)] <- ret$contribution[nrow(ret)]

  ret$variable_name <- ret$variable
  ret$variable_name <- factor(ret$variable_name, levels=c(ret$variable_name, ''))
  ret$variable_name[nrow(ret)] <- ''

  ret$variable_value <- '' # column for consistency

  ret
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
  is_numeric_not_int <- function(...){(is.numeric(...) & !is.integer(...)) | is.complex(...)}

  # takes average / median of columns

  # (numeric not integer) or complex
  numeric_cols <- unlist(lapply(subset, is_numeric_not_int))
  numeric_cols <- names(numeric_cols[numeric_cols == TRUE])
  if(length(numeric_cols) == 1){
    df_numeric <- data.frame(tmp = mean(subset[,numeric_cols]))
    colnames(df_numeric) <- numeric_cols[1]
  } else {
    df_numeric <- t(as.data.frame(colMeans(subset[,numeric_cols])))
  }

  # integer
  int_cols <- unlist(lapply(subset, is.integer))
  int_cols <- names(int_cols[int_cols == TRUE])
  df_int <- as.data.frame(lapply(int_cols, function(col) {
    tab <- table(subset[,col])
    tab_val <- attr(tab, 'dimnames')[[1]]
    tab_val <- tab_val[which.max(tab)]
    as.integer(tab_val)
  }), stringsAsFactors = FALSE)
  colnames(df_int) <- int_cols

  # logical
  logical_cols <- unlist(lapply(subset, is.logical))
  logical_cols <- names(logical_cols[logical_cols == TRUE])
  df_logical <- as.data.frame(lapply(logical_cols, function(col) {
    tab <- table(subset[,col])
    tab_val <- attr(tab, 'dimnames')[[1]]
    tab_val <- tab_val[which.max(tab)]
    as.logical(tab_val)
  }), stringsAsFactors = FALSE)
  colnames(df_logical) <- logical_cols

  # factors
  factor_cols <- unlist(lapply(subset, is.factor))
  factor_cols <- names(factor_cols[factor_cols == TRUE])
  df_factory <- as.data.frame(lapply(factor_cols, function(col) {
    factor(names(which.max(table(subset[,col]))), levels = levels(subset[,col]))
  }))
  colnames(df_factory) <- factor_cols

  # character
  other_cols <- unlist(lapply(subset, is.character))
  other_cols <- names(other_cols[other_cols == TRUE])
  df_others <- as.data.frame(lapply(other_cols, function(col) {
    tab <- table(subset[,col])
    tab_names <- attr(tab, 'dimnames')[[1]]
    tab_names[which.max(tab)]
  }), stringsAsFactors = FALSE)
  colnames(df_others) <- other_cols

  outs <- list()
  if(!ncol(df_numeric) == 0){outs <- append(list(df_numeric), outs)}
  if(!ncol(df_int) == 0){outs <- append(list(df_int), outs)}
  if(!ncol(df_logical) == 0){outs <- append(list(df_logical), outs)}
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
