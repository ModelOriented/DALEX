# internal function
# combine explainers for the plot() functions
# if more than one explainer is provided,
# then this function merge explainers into a single dataframe
combine_explainers <- function(x, ...) {
  df <- x
  df$label <- as.character(df$label)
  all_labels <- unique(df$label)
  class(df) <- "data.frame"
  df$name <- seq.int(nrow(df))
  dfl <- list(...)
  if (length(dfl) > 0) {
    for (resp in dfl) {
      class(resp) <- "data.frame"
      resp$label <- as.character(resp$label)
      resp$name <- seq.int(nrow(resp))

      #if labels are duplicated, fix it
      if (any(unique(resp$label) %in% all_labels)) {
        old_labels_corrected <- unique(resp$label)
        all_labels <- make.unique(c(all_labels, old_labels_corrected))
        all_labels_corrected <- tail(all_labels, length(old_labels_corrected))
        for (i in seq_along(old_labels_corrected)) {
          resp[resp[,"label"] == old_labels_corrected[i], "label"] <- all_labels_corrected[i]
         }
      }
      all_labels <- c(all_labels, unique(resp$label))
      df <- rbind(df, resp)
    }
  }
  df
}

# test explainer
# test if the explainer object has all reqired fields
test_expaliner <- function(explainer,
                           has_data = FALSE,
                           has_y = FALSE,
                           function_name = "variable_profile") {
  # run checks against the explainer objects
  if (!("explainer" %in% class(explainer)))
       stop(paste0("The ",function_name," function requires an object created with explain() function."))
  if (has_data && is.null(explainer$data))
    stop(paste0("The ",function_name," function requires explainers created with specified 'data' parameter."))
  if (has_y && is.null(explainer$y))
    stop(paste0("The ",function_name," function requires explainers created with specified 'y' parameter."))

  TRUE
}
