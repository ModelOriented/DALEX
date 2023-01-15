#' Plot Generic for Break Down Objects
#'
#' Displays a waterfall aggregated shap plot for objects of \code{shap_aggregated} class.
#'
#' @param x an explanation object created with function \code{\link[DALEX]{explain}}.
#' @param ... other parameters like \code{vcolors}, \code{vnames}, \code{min_max}, \code{digits}, \code{rounding_function}, \code{baseline}, \code{subtitle}, \code{baseline}, \code{max_vars}.
#' @param max_features  maximal number of features to be included in the plot. default value is \code{10}.
#' @param add_contributions if \code{TRUE}, variable contributions will be added to the plot
#' @param add_boxplots if \code{TRUE}, boxplots of SHAP will be shown
#' @param shift_contributions number describing how much labels should be shifted to the right, as a fraction of range. By default equal to \code{0.05}.
#' @param title a character. Plot title. By default \code{"Break Down profile"}.
#'
#' @return a \code{ggplot2} object.
#'
#' @import ggplot2
#' @importFrom utils tail
#'
#' @examples
#' library("DALEX")
#' set.seed(1313)
#' model_titanic_glm <- glm(survived ~ gender + age + fare,
#'                        data = titanic_imputed, family = "binomial")
#' explain_titanic_glm <- explain(model_titanic_glm,
#'                            data = titanic_imputed,
#'                            y = titanic_imputed$survived,
#'                            label = "glm")
#'
#' \donttest{
#' bd_glm <- shap_aggregated(explain_titanic_glm, titanic_imputed[1:10, ])
#' bd_glm
#' plot(bd_glm)
#' plot(bd_glm, max_features = 3)
#' plot(bd_glm, max_features = 3,
#'      vnames = c("average","+ male","+ young","+ cheap ticket", "+ other factors", "final"))
#' }
#'
#' @export
plot.shap_aggregated <- function(x, ..., shift_contributions = 0.05, add_contributions = TRUE, add_boxplots = TRUE, max_features = 10, title = "Aggregated SHAP") {
  x <- select_only_k_features(x, k = max_features)
  aggregate <- x[[1]]
  raw <- x[[2]]
  class(aggregate) <- c('break_down', class(aggregate))

  # keep names to avoid errors during the check()
  right_side <- position <- mean_boxplot <- contribution <- NULL

  # ret has at least 3 columns: first and last are intercept and prediction
  aggregate$mean_boxplot <- c(0, aggregate$cumulative[1:(nrow(aggregate)-2)], 0)
  raw <- merge(x = as.data.frame(aggregate[,c('variable', 'position', 'mean_boxplot')]), y = raw, by.x = "variable", by.y = "variable_name", all.y = TRUE)

  # max_features = max_features + 1 because we have one more class already - "+ all other features"
  p <- plot(aggregate, ..., add_contributions = FALSE, max_features = max_features + 1, title = title)

  if(add_boxplots){
    p <- p + geom_boxplot(data = raw,
                          aes(y = contribution + mean_boxplot,
                              x = position + 0.5,
                              group = position,
                              fill = "#371ea3",
                              xmin = min(contribution) - 0.85,
                              xmax = max(contribution) + 0.85),
                          color = "#371ea3",
                          fill = "#371ea3",
                          width = 0.15)
  }

  if (add_contributions) {
    aggregate$right_side <- pmax(aggregate$cumulative,  aggregate$cumulative - aggregate$contribution)
    drange <- diff(range(aggregate$cumulative))

    p <- p + geom_text(aes(y = right_side),
                       vjust = -1,
                       nudge_y = drange*shift_contributions,
                       hjust = -0.2,
                       color = "#371ea3")
  }

  p
}

select_only_k_features <- function(input, k = 10) {
  x <- input[[1]]
  y <- input[[2]]

  # filter-out redundant rows
  contribution_sum <- tapply(x$contribution, x$variable_name, function(contribution) sum(abs(contribution), na.rm = TRUE))
  contribution_ordered_vars <- names(sort(contribution_sum[!(names(contribution_sum) %in% c("", "intercept"))]))
  variables_keep <- tail(contribution_ordered_vars, k)
  variables_remove <- setdiff(contribution_ordered_vars, variables_keep)

  if (length(variables_remove) > 0) {
    x_remove   <- x[x$variable_name %in% variables_remove,]
    x_keep     <- x[!(x$variable_name %in% c(variables_remove, "")),]
    x_prediction <- x[x$variable == "prediction",]
    row.names(x_prediction) <- x_prediction$label
    remainings <- tapply(x_remove$contribution, x_remove$label, sum, na.rm=TRUE)
    # fix position and cumulative in x_keep
    x_keep$position <- as.numeric(as.factor(x_keep$position)) + 2
    for (i in 1:nrow(x_keep)) {
      if (x_keep[i,"variable_name"] == "intercept") {
        x_keep[i,"cumulative"] <- x_keep[i,"contribution"]
      } else {
        x_keep[i,"cumulative"] <- x_keep[i - 1,"cumulative"] + x_keep[i,"contribution"]
      }
    }
    # for each model we shall calculate the others statistic
    x_others <- data.frame(variable = "+ all other factors",
                           contribution = remainings,
                           variable_name = "+ all other factors",
                           variable_value = "",
                           cumulative = x_prediction[names(remainings),"cumulative"],
                           sign = sign(remainings),
                           position = 2,
                           label = names(remainings))
    #
    x <- rbind(x_keep, x_others, x_prediction)
    y$variable_name <- factor(ifelse(y$variable_name %in% variables_remove, "+ all other factors", as.character(y$variable_name)), levels = levels(x$variable_name))
  }

  list(aggregated = x, raw = y)
}
