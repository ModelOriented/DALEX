#' Plot Model Performance Explanations
#'
#' @param x a model to be explained, preprocessed by the \code{\link{explain}} function
#' @param ... other parameters
#' @param geom either \code{"ecdf"} or \code{"boxplot"} determines how residuals shall be summarized
#' @param lossFunction function that calculates the loss for a model based on model residuals. By default it's the root mean square.
#' @param show_outliers number of largest residuals to be presented (only when geom = boxplot).
#' @param ptlabel either \code{"name"} or \code{"index"} determines the naming convention of the outliers
#'
#' @return An object of the class \code{model_performance_explainer}.
#'
#' @export
#' @examples
#'  \dontrun{
#' library("ranger")
#' titanic_ranger_model <- ranger(survived~., data = titanic_imputed, num.trees = 50,
#'                                probability = TRUE)
#' explainer_ranger  <- explain(titanic_ranger_model, data = titanic_imputed[,-8],
#'                              y = titanic_imputed$survived)
#' mp_ranger <- model_performance(explainer_ranger)
#' plot(mp_ranger)
#' plot(mp_ranger, geom = "boxplot", show_outliers = 1)
#'
#' titanic_ranger_model2 <- ranger(survived~gender + fare, data = titanic_imputed,
#'                                 num.trees = 50, probability = TRUE)
#' explainer_ranger2  <- explain(titanic_ranger_model2, data = titanic_imputed[,-8],
#'                               y = titanic_imputed$survived,
#'                               label = "ranger2")
#' mp_ranger2 <- model_performance(explainer_ranger2)
#' plot(mp_ranger, mp_ranger2)
#'
#' titanic_glm_model <- glm(survived~., data = titanic_imputed, family = "binomial")
#' explainer_glm <- explain(titanic_glm_model, data = titanic_imputed[,-8],
#'                          y = titanic_imputed$survived, label = "glm",
#'                     predict_function = function(m,x) predict.glm(m,x,type = "response"))
#' mp_glm <- model_performance(explainer_glm)
#' plot(mp_glm)
#'
#' titanic_lm_model <- lm(survived~., data = titanic_imputed)
#' explainer_lm <- explain(titanic_lm_model, data = titanic_imputed[,-8],
#'                         y = titanic_imputed$survived, label = "lm")
#' mp_lm <- model_performance(explainer_lm)
#' plot(mp_lm)
#'
#' plot(mp_ranger, mp_glm, mp_lm)
#' plot(mp_ranger, mp_glm, mp_lm, geom = "boxplot")
#' plot(mp_ranger, mp_glm, mp_lm, geom = "boxplot", show_outliers = 1)
#'  }
#'
plot.model_performance_explainer <- function(x, ..., geom = "ecdf", show_outliers = 0, ptlabel = "name", lossFunction = function(x) sqrt(mean(x^2))) {
  if (!(ptlabel %in% c("name", "index"))){
    stop("The plot.model_performance() function requires label to be name or index.")
  }
  # extract residuals
  if (length(list(...)) == 0) {
    # if single explainer
    df <- x$residuals
  } else {
    # if multiple explainers
    args <- lapply(list(...),
                   function(tmp) tmp$residuals)
    args[["x"]] <- x$residuals
    df <- do.call(combine_explainers, rev(args))
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
      scale_color_manual(name = "Model", values = colors_discrete_drwhy(nlabels)) +
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
      scale_fill_manual(name = "Model", values = colors_discrete_drwhy(nlabels)) +
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
