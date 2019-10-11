#' Update label of explainer object
#'
#' Function allows users to update label of any explainer in a unified way. It doesn't require knowledge about structre of an explainer.
#'
#' @param explainer - explainer object that is supposed to be updated.
#' @param new_label - new label, is going to be passed to an explainer
#'
#' @return updated explainer object
#'
#' @examples
#' aps_lm_model4 <- lm(m2.price ~., data = apartments)
#' aps_lm_explainer4 <- explain(aps_lm_model4, data = apartments, label = "model_4v")
#' new_explainer <- update_label(aps_lm_explainer4, new_label = "new_lm")
#' @rdname update_label
#' @export

update_label <- function(explainer, new_label) {
  if (!is.character(new_label)) {
    stop("new_label parameter has to be a character")
  }
  if (!"explainer" %in% class(explainer)) {
    stop("explainer parameter has to be an explainer object")
  }
  explainer$label <- new_label
  explainer
}

#' Update data of an explainer object
#'
#' Function allows users to update data an y of any explainer in a unified way. It doesn't require knowledge about structre of an explainer.
#'
#' @param explainer - explainer object that is supposed to be updated.
#' @param new_data - new data, is going to be passed to an explainer
#' @param new_y - new y, is going to be passed to an explainer
#'
#' @return updated explainer object
#'
#' @examples
#' aps_lm_model4 <- lm(m2.price ~., data = apartments)
#' aps_lm_explainer4 <- explain(aps_lm_model4, data = apartments, label = "model_4v")
#' new_explainer <- update_data(aps_lm_explainer4, new_data = apartmentsTest, new_y = apartmentsTest$m2.price)
#'
#' @rdname update_data
#' @export

update_data <- function(explainer, new_data, new_y) {
  if (all(!(c("data.frame", "tbl")) %in% class(new_data))) {
    stop("new_data has to be a data.frame or tibble")
  }
  if (!"explainer" %in% class(explainer)) {
    stop("expaliner parameter has to be an explainer object")
  }
  if (is.factor(new_y)) {
    message("Please note that new_y is a factor. Consider changing the 'y' to a logical or numerical vector.")
  }
  if (is.data.frame(new_y)) {
    new_y <- unlist(new_y, use.names = FALSE)
  }

  explainer$data <- new_data
  explainer$y <- new_y
  explainer
}
