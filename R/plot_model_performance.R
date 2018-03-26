#' Model Performance Plots
#'
#' @param x a model to be explained, preprocessed by the 'explain' function
#' @param ... other parameters
#'
#' @return An object of the class 'model_performance_explainer'.
#'
#' @export
#' @examples
#' #\dontrun{
#' library("breakDown")
#' library("randomForest")
#' HR_rf_model <- randomForest(left~., data = breakDown::HR_data, ntree = 100)
#' explainer_rf  <- explain(HR_rf_model, data = HR_data, y = HR_data$left)
#' mp_rf <- model_performance(explainer_rf)
#' plot(mp_rf)
#'
#' HR_glm_model <- glm(left~., data = breakDown::HR_data, family = "binomial")
#' explainer_glm <- explain(HR_glm_model, data = HR_data, y = HR_data$left, label = "glm",
#'                     predict_function = function(m,x) predict.glm(m,x,type = "response"))
#' mp_glm <- model_performance(explainer_glm)
#' plot(mp_glm)
#'
#' HR_lm_model <- lm(left~., data = breakDown::HR_data)
#' explainer_lm <- explain(HR_lm_model, data = HR_data, y = HR_data$left)
#' mp_lm <- model_performance(explainer_lm)
#' plot(mp_lm)
#'
#' plot(mp_rf, mp_glm, mp_lm)
#' #}
#'
plot.model_performance_explainer <- function(x, ...) {
  df <- x
  class(df) <- "data.frame"

  dfl <- list(...)
  if (length(dfl) > 0) {
    for (resp in dfl) {
      class(resp) <- "data.frame"
      df <- rbind(df, resp)
    }
  }
  label <- NULL
  ggplot(df, aes(abs(diff), color = label)) +
    stat_ecdf(geom = "step") +
    stat_ecdf(geom = "point") +
    theme_mi2() +
    scale_color_brewer(name = "Model", type = "qual", palette = "Dark2") +
    xlab("| residuals |") +
    scale_y_continuous(breaks = seq(0,1,0.1),
                       labels = paste(seq(100,0,-10),"%"),
                       trans = "reverse",
                       name = "") +
    ggtitle("Ecdf of | residuals |")

}
