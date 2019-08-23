#' Prints Natural Language Descriptions
#'
#' @param x an individual explainer produced with the `describe()` function
#' @param ... other arguments
#'
#' @export
print.description <- function(x, ...) {
  for (element in x) {
    cat(element, "\n")
  }

  return(invisible(NULL))
}
