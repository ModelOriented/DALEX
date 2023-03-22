transform_shap_to_break_down <- function(shaps, label, intercept, prediction, order_of_variables=NULL, agg=FALSE) {
  ret <- as.data.frame(shaps)
  ret <- ret[, c('variable', 'label', 'contribution')]
  ret$variable <- as.character(ret$variable)
  rownames(ret) <- ret$variable

  if(!is.null(order_of_variables) && is.vector(order_of_variables)){
    ret <- ret[order_of_variables,]
  }

  ret$position <- (nrow(ret) + 1):2
  ret$sign <- ifelse(ret$contribution >= 0, "1", "-1")

  ret <- rbind(ret, data.frame(variable = "intercept",
                               label = label,
                               contribution = intercept,
                               position = max(ret$position) + 1,
                               sign = "X"),
               make.row.names=FALSE)

  ret <- rbind(ret, data.frame(variable = "prediction",
                               label = label,
                               contribution = prediction,
                               position = 1,
                               sign = "X"),
               make.row.names=FALSE)

  ret <- ret[order(ret$position, decreasing = TRUE), ]

  ret$cumulative <- cumsum(ret$contribution)
  ret$cumulative[nrow(ret)] <- ret$contribution[nrow(ret)]

  ret$variable_name <- c('intercept', as.data.frame(shaps)[, c('variable_name')], '')
  ret$variable_name <- factor(ret$variable_name, levels=ret$variable_name)
  ret$variable_name[nrow(ret)] <- ''

  if(agg) {
    ret$variable_value <- ''
  } else {
    ret$variable_value <- c(1, as.data.frame(shaps)[, c('variable_name')], '')
  }


  class(ret) <- c('predict_parts', 'break_down', 'data.frame')

  ret
}

kernelshap_to_shap <- function(ks, new_observation, explainer, agg=FALSE) {
  res <- as.data.frame(t(ks$S))

  colnames(res) <- c('contribution')
  res$variable_name <- rownames(res)

  if(agg) {
    res$variable_value <- ''
  } else {
    res$variable_value <- unname(unlist(new_observation))
  }

  res$variable <- paste0(res$variable_name, ' = ', nice_format(res$variable_value))
  res$sign <- ifelse(res$contribution > 0, 1, -1)
  res$label <- explainer$label
  res$B <- 0

  attr(res, "prediction") <- as.numeric(
    explainer$predict_function(explainer$model, new_observation)
  )
  attr(res, "intercept") <- as.numeric(ks$baseline)

  class(res) <- c('predict_parts', 'shap', 'break_down_uncertainty', 'data.frame')
  res
}

nice_format <- function(x) {
  if (is.numeric(x)) {
    as.character(signif(x, 4))
  } else if ("tbl" %in% class(x)) {
    as.character(x[[1]])
  } else {
    as.character(x)
  }
}
