#' Prints scikitlearn_model class
#' 
#' @param x a model created with `scikitlearn_model` function
#' @param ... other arguments
#' 
#' @examples 
#' library("DALEX")
#' library("reticulate")
#' have_sklearn <- reticulate::py_module_available("sklearn.ensemble")
#' if(have_sklearn) {
#'   model <- scikitlearn_model(system.file("extdata", "gbm.pkl", package = "DALEX")) 
#'   print(model)
#' } else {
#'   print('Python testing environment is required.')
#' }
#' @export
 
print.scikitlearn_model <- function(x, ...){
  cat("Model name: ", x$label, "\n")
  cat("Model type: ", x$type, "\n")
  return(invisible(NULL))
}
