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
