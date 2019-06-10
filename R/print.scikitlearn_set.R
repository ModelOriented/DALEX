#' Prints scikitlearn_set class
#' 
#' @param x a model created with `scikitlearn_model` function
#' @param ... other arguments
#' @examples 
#' library("DALEX")
#' library("reticulate")
#' have_sklearn <- reticulate::py_module_available("sklearn.ensemble")
#' if(have_sklearn) {
#'   model <- scikitlearn_model(system.file("extdata", "gbm.pkl", package = "DALEX")) 
#'   print(model$params)
#' } else {
#'   print('Python testing environment is required.')
#' }
#' @export


print.scikitlearn_set <- function(x, ...){
  for (i in 1:length(x)) {
    if(is.null(x[[i]])){
      cat(names(x[i]), ": ", "NULL", " \n", sep = "")
    }else{
      cat(names(x[i]), ": ", x[[i]], " \n", sep = "")
    }
  }
  return(invisible(NULL))
}
