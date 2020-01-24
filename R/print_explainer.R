#' Print Explainer Summary
#'
#' @param x a model explainer created with the `explain` function
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
#' library("ranger")
#' titanic_ranger_model <- ranger(survived~., data = titanic_imputed, num.trees = 50,
#'                                probability = TRUE)
#' explainer_ranger  <- explain(titanic_ranger_model, data = titanic_imputed[,-8],
#'                              y = titanic_imputed$survived,
#'                              label = "model_ranger")
#' explainer_ranger
#'  }
#'
print.explainer <- function(x, ...) {
  cat("Model label: ", x$label, "\n")
  cat("Model class: ", paste(x$class, collapse = ","), "\n")
  cat("Data head  :\n")
  print(head(x$data,2))
  return(invisible(NULL))
}

