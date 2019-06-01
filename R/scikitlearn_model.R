#' Wrapper for Python scikit-learn models
#' 
#' scikit-learn models may be lodaed into R enviromente like any other Python object. We may need it to inspect performance of our model 
#' and compre it with others, using R tools like DALEX. This function creates object that is easy accessible version of model 
#' exported form python via pickle file.
#' 
#' @usage scikitlearn_model("gbm.pkl")
#' 
#' @param path string - a path to pickle file
#' 
#' 
#' @return An object of the class 'scikitlearn_model'.
#'
#' It's a list with following fields:
#'
#' \itemize{
#' \item \code{name} name of model
#' \item \code{type} type model, classification or regression
#' \item \code{params} object of class `scikitlearn_set` which in fact is list that consist of parameters of our model.
#' \item \code{predict_function} predict function extracted from original model. It is adjusted to DALEX demands and therfore fully compatibile. 
#' \item \code{model} it is original model received vie reticiulate function. Use it for computation.
#' }
#' 
#' \bold{Example of Python code}\cr
#' \code{
#' from pandas import DataFrame, read_csv \cr
#' import pandas as pd\cr
#' import pickle\cr
#' import sklearn.ensemble\cr
#' model = sklearn.ensemble.GradientBoostingClassifier() \cr
#' model = model.fit(titanic_train_X, titanic_train_Y)\cr
#' pickle.dump(model, open("gbm.pkl", "wb"))\cr
#' }
#' 
#' @examples 
#' ##usage with explain()
#' have_picke <- reticulate::py_module_available("pickle")
#' ## Not run:
#' 
#' if(have_picke){
#' library(dplyr)
#' library(DALEX)
#' library(reticulate)
#' 
#' explainer <- scikitlearn_model(".//inst//gbm.pkl") %>% DALEX::explain(data = titanic_test, y = titanic_test$survived)
#' model_performance(explainer)
#' }else{
#' print('Python testing environment is required.')
#' }
#' 
#' 
#' ## End(Not run)
#' 
#' ## Predictions with nedata
#' ## Not run:
#' if(have_pickle){
#' library(DALEX)
#' library(reticulate)
#' model <- scikitlearn_model(".//inst//gbm.pkl")
#' predictions <- model$predict_function(model$model, titanic_test_X)
#' }else{
#' print('Python testing environment is required.')
#' }
#' ## End(Not run)
#' 
#' @rdname scikitlearn_model
#' @export
#' 
scikitlearn_model <- function(path){
  # loaded reticiulate library requaired. There is no dependency
  model <- reticulate::py_load_object(path)
  # params are represented as one longe string
  params <- model$get_params
  # taking first element since strsplit() returns list of vectors
  params <- strsplit(as.character(params), split = ",")[[1]]
  # replacing blanks and other signs that we don't need and are pasted with params names
  params <- gsub(params, pattern = "\n", replacement = "", fixed = TRUE) 
  params <- gsub(params, pattern = " ", replacement = "", fixed = TRUE)
  # splitting after "=" mark and taking first element (head(n = 1L)) provides as with params names
  params <- lapply(strsplit(params, split = "="), head, n = 1L)
  # removing name of function from the first parameter
  params[[1]] <- strsplit(params[[1]], split = "\\(" )[[1]][2]
  # setting freshly extracted parameters names as labels for list
  names(params) <- as.character(params)
  #extracting parameters value
  params <- lapply(params, function(x){
    model[[x]]
  })
  
  class(params) <- "scikitlearn_set"
  
  #name of our model
  name <- strsplit(as.character(model), split = "\\(")[[1]][1]
  
  if("predict_proba" %in% names(model)){
    predict_function <- function(model, newdata){
      # we take second cloumn which indicates probability of `1` to adapt to DALEX predict functions (yhat)
      model$predict_proba(newdata)[, 2]
    }
    type = "classification"
  }else{
    predict_function <- function(model, newdata){
      model$predict(newdata)
    }
    type = "regression"
  }
  
  scikitlearn_model <- list(
    name = name,
    type = type,
    params = params,
    predict_function = predict_function,
    model = model
  )
  class(scikitlearn_model) <- "scikitlearn_model"
  scikitlearn_model
  
}



