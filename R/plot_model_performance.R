#' Model Performance Plots
#'
#' @param x a model to be explained, preprocessed by the 'explain' function
#' @param ... other parameters
#' @param geom either \code{"ecdf"} or \code{"boxplot"} determines how residuals shall be summarized
#' @param lossFunction function that calculates the loss for a model based on model residuals. By default it's the root mean square.
#' @param show_outliers number of largest residuals to be presented (only when geom = boxplot).
#' @param ptlabel either \code{"name"} or \code{"index"} determines the naming convention of the outliers
#'
#' @return An object of the class 'model_performance_explainer'.
#'
#' @export
#' @examples
#'  \dontrun{
#' library("randomForest")
#' HR_rf_model <- randomForest(status == "fired"~., data = HR, ntree = 100)
#' explainer_rf  <- explain(HR_rf_model, data = HR, y = HR$status == "fired")
#' mp_rf <- model_performance(explainer_rf)
#' plot(mp_rf)
#' plot(mp_rf, geom = "boxplot", show_outliers = 1)
#'
#' HR_glm_model <- glm(status == "fired"~., data = HR, family = "binomial")
#' explainer_glm <- explain(HR_glm_model, data = HR, y = HR$status == "fired", label = "glm",
#'                     predict_function = function(m,x) predict.glm(m,x,type = "response"))
#' mp_glm <- model_performance(explainer_glm)
#' plot(mp_glm)
#'
#' HR_lm_model <- lm(status == "fired"~., data = HR)
#' explainer_lm <- explain(HR_lm_model, data = HR, y = HR$status == "fired")
#' mp_lm <- model_performance(explainer_lm)
#' plot(mp_lm)
#'
#' plot(mp_rf, mp_glm, mp_lm)
#' plot(mp_rf, mp_glm, mp_lm, geom = "boxplot")
#' plot(mp_rf, mp_glm, mp_lm, geom = "boxplot", show_outliers = 1)
#'  }
#'
plot.model_performance_explainer <- function(x, ..., geom = "ecdf", show_outliers = 0, ptlabel = "name", lossFunction = function(x) sqrt(mean(x^2))) {
  if (!(ptlabel %in% c("name", "index"))){
    stop("The plot.model_performance() function requires label to be name or index.")
  }
  df <- x
  class(df) <- "data.frame"
  df$name <- seq.int(nrow(df))
  dfl <- list(...)
  if (length(dfl) > 0) {
    for (resp in dfl) {
      class(resp) <- "data.frame"
      resp$name <- seq.int(nrow(resp))
      df <- rbind(df, resp)
    }
  }
  df$label <- reorder(df$label, df$diff, lossFunction)
  label <- name <- NULL
  if (ptlabel == "name") {
    df$name <- NULL
    df$name <- rownames(df)
  }
  nlabels <- length(unique(df$label))
  if (geom == "ecdf") {
    pl <-   ggplot(df, aes(abs(diff), color = label)) +
      stat_ecdf(geom = "step") +
      theme_drwhy() +
      scale_color_manual(name = "Model", values = theme_drwhy_colors(nlabels)) +
      xlab(expression(group("|", residual, "|"))) +
      scale_y_continuous(breaks = seq(0,1,0.1),
                         labels = paste(seq(100,0,-10),"%"),
                         trans = "reverse",
                         name = "") +
      ggtitle(expression(paste("Distribution of ", group("|", residual, "|"))))
  } else {
    pl <- ggplot(df, aes(x = label, y = abs(diff), fill = label)) +
      stat_boxplot(alpha = 0.4, coef = 1000) +
      stat_summary(fun.y = lossFunction, geom="point", shape = 20, size=10, color="red", fill="red") +
      theme_drwhy_vertical() +
      scale_fill_manual(name = "Model", values = theme_drwhy_colors(nlabels)) +
      ylab("") + xlab("") +
      ggtitle(
        expression(paste("Boxplots of ", group("|", residual, "|"))),
        "Red dot stands for root mean square of residuals"
      ) +
      coord_flip()
    if (show_outliers > 0) {
      df$rank <- unlist(tapply(-abs(df$diff), df$label, rank, ties.method = "min"))
      df_small <- df[df$rank <= show_outliers,]
      pl <- pl +
        geom_point(data = df_small) +
        geom_text(data = df_small,
                  aes(label = name), srt = 90,
                  hjust = -0.2, vjust = 1)
    }
  }
  pl
}
