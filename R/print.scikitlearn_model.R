#' Prints scikitlearn_model class
#' 
#' @param x a model created with `scikitlearn_model` function
#' @param ... other arguments
#' 
#' @export
#' 
#' 
print.scikitlearn_model <- function(x, ...){
  cat("Model name: ", x$name, "\n")
  cat("Model type: ", x$type, "\n")
  return(invisible(NULL))
}
