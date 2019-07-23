#' Prints Explainer Summary
#'
#' @param x a model expaliner created with the `explain` function
#' @param ... other parameters
#'
#' @export
#' @import ggplot2
#'
#' @examples
#'
#' aps_lm_model4 <- lm(m2.price~., data = apartments)
#' aps_lm_explainer4 <- explain(aps_lm_model4, data = apartments, y = apartments$m2.price, 
#'                              label = "model_4v")
#' aps_lm_explainer4
#'
#'  \dontrun{
#' library("randomForest")
#' HR_rf_model4 <- randomForest(as.factor(status == "fired")~., data = HR, ntree = 100)
#' HR_rf_explainer4 <- explain(HR_rf_model4, data = HR, label = "model_rf")
#' HR_rf_explainer4
#'  }
#'
print.explainer <- function(x, ...) {
  cat("Model label: ", x$label, "\n")
  cat("Model class: ", paste(x$class, collapse = ","), "\n")
  cat("Data head  :\n")
  print(head(x$data,2))
  return(invisible(NULL))
}

