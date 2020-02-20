#' Update label of explainer object
#'
#' Function allows users to update label of any explainer in a unified way. It doesn't require knowledge about structre of an explainer.
#'
#' @param explainer - explainer object that is supposed to be updated.
#' @param label - new label, is going to be passed to an explainer
#' @param verbose - logical, indicates if information about update should be printed
#'
#' @return updated explainer object
#'
#' @examples
#' aps_lm_model4 <- lm(m2.price ~., data = apartments)
#' aps_lm_explainer4 <- explain(aps_lm_model4, data = apartments, label = "model_4v")
#' explainer <- update_label(aps_lm_explainer4, label = "lm")
#' @rdname update_label
#' @export

update_label <- function(explainer, label, verbose = TRUE) {
  if (!is.character(label)) {
    stop("label parameter has to be a character")
  }
  if (!"explainer" %in% class(explainer)) {
    stop("explainer parameter has to be an explainer object")
  }
  explainer$label <- label
  verbose_cat("  -> model label       : ", label, "\n", verbose = verbose)
  verbose_cat("",color_codes$green_start,"An explainer has been updated!",color_codes$green_end,"\n", verbose = verbose)

  explainer
}

#' Update data of an explainer object
#'
#' Function allows users to update data an y of any explainer in a unified way. It doesn't require knowledge about structre of an explainer.
#'
#' @param explainer - explainer object that is supposed to be updated.
#' @param data - new data, is going to be passed to an explainer
#' @param y - new y, is going to be passed to an explainer
#' @param verbose - logical, indicates if information about update should be printed
#'
#' @return updated explainer object
#'
#' @examples
#' aps_lm_model4 <- lm(m2.price ~., data = apartments)
#' aps_lm_explainer4 <- explain(aps_lm_model4, data = apartments, label = "model_4v")
#' explainer <- update_data(aps_lm_explainer4, data = apartmentsTest, y = apartmentsTest$m2.price)
#'
#' @rdname update_data
#' @export

update_data <- function(explainer, data, y = NULL, verbose = TRUE) {
  if (all(!(c("data.frame", "tbl")) %in% class(data))) {
    stop("data has to be a data.frame or tibble")
  }
  if (!"explainer" %in% class(explainer)) {
    stop("expaliner parameter has to be an explainer object")
  }
  explainer$data <- data
  verbose_cat("  -> data              : ", nrow(data), " rows ", ncol(data), " cols \n", verbose = verbose)
  if (!is.null(y)){
    if (is.factor(y)) {
      message("Please note that y is a factor. Consider changing the 'y' to a logical or numerical vector.")
    }
    if (is.data.frame(y)) {
      y <- unlist(y, use.names = FALSE)
    }
    verbose_cat("  -> target variable   : ", length(y), " values \n", verbose = verbose)
    explainer$y <- y
  }
  if (!is.null(explainer$predict_function)){
    explainer$y_hat <- explainer$predict_function(explainer$model, data)
    explainer$residuals <- explainer$y - explainer$predict_function(explainer$model, data)
  }
  verbose_cat("",color_codes$green_start,"An explainer has been updated!",color_codes$green_end,"\n", verbose = verbose)
  explainer
}
