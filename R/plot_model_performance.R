#' Plot Dataset Level Model Performance Explanations
#'
#' @param x a model to be explained, preprocessed by the \code{\link{explain}} function
#' @param ... other parameters
#' @param geom either \code{"prc"}, \code{"roc"}, \code{"ecdf"}, \code{"boxplot"}, \code{"gain"}, \code{"lift"} or \code{"histogram"} determines how residuals shall be summarized
#' @param loss_function function that calculates the loss for a model based on model residuals. By default it's the root mean square. NOTE that this argument was called \code{lossFunction}.
#' @param lossFunction alias for \code{loss_function} held for backwards compatibility.
#' @param show_outliers number of largest residuals to be presented (only when geom = boxplot).
#' @param ptlabel either \code{"name"} or \code{"index"} determines the naming convention of the outliers
#'
#' @return An object of the class \code{model_performance}.
#'
#' @export
#' @examples
#'  \donttest{
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
#' plot(mp_ranger, mp_ranger2, geom = "prc")
#' plot(mp_ranger, mp_ranger2, geom = "roc")
#' plot(mp_ranger, mp_ranger2, geom = "lift")
#' plot(mp_ranger, mp_ranger2, geom = "gain")
#' plot(mp_ranger, mp_ranger2, geom = "boxplot")
#' plot(mp_ranger, mp_ranger2, geom = "histogram")
#' plot(mp_ranger, mp_ranger2, geom = "ecdf")
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
#' }
#'
#
plot.model_performance <- function(x, ..., geom = "ecdf", show_outliers = 0, ptlabel = "name", lossFunction = loss_function, loss_function = function(x) sqrt(mean(x^2))) {
  if (!(ptlabel %in% c("name", "index"))){
    stop("The plot.model_performance() function requires label to be name or index.")
  }

  # lossFunction is deprecated
#  if (methods::hasArg("lossFunction")) {
#    warning("lossFunction is deprecated, please use loss_function instead")
#    loss_function <- list(...)[["lossFunction"]]
#  }

  # extract residuals
  # combine into a single object
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

  df$label <- reorder(df$label, df$diff, loss_function)
  # Sort df based on the label's levels to make sure downstream functions relying
  # on the rows order work appropriately
  df <- df[order(df$label), ]
  if (ptlabel == "name") {
    df$name <- NULL
    df$name <- rownames(df)
  }
  nlabels <- length(unique(df$label))
  switch(geom,
           ecdf = plot.model_performance_ecdf(df, nlabels),
           boxplot = plot.model_performance_boxplot(df, show_outliers, loss_function, nlabels),
           histogram = plot.model_performance_histogram(df, nlabels),
           prc = plot.model_performance_prc(df, nlabels),
           roc = plot.model_performance_roc(df, nlabels),
           gain = plot.model_performance_gain(df, nlabels),
           lift = plot.model_performance_lift(df, nlabels)
  )
}


plot.model_performance_ecdf <- function(df, nlabels) {
  label <- name <- NULL
  ggplot(df, aes(abs(diff), color = label)) +
    stat_ecdf(geom = "step") +
    theme_default_dalex() +
    scale_color_manual(name = "Model", values = colors_discrete_drwhy(nlabels)) +
    xlab(expression(group("|", residual, "|"))) +
    scale_y_continuous(breaks = seq(0,1,0.1),
                       labels = paste(seq(100,0,-10),"%"),
                       trans = "reverse",
                       name = "") +
    ggtitle(expression(paste("Reverse cumulative distribution of ", group("|", residual, "|"))))
}

plot.model_performance_boxplot <- function(df, show_outliers, loss_function, nlabels) {
  label <- name <- NULL
  pl <- ggplot(df, aes(x = label, y = abs(diff), fill = label)) +
    stat_boxplot(alpha = 0.4, coef = 1000) +
    stat_summary(fun = loss_function, geom = "point", shape = 20, size=10, color="red", fill="red") +
    theme_vertical_default_dalex() +
    scale_fill_manual(name = "Model", values = colors_discrete_drwhy(nlabels)) +
    ylab("") +
    scale_x_discrete("", limits = rev(levels(df$label))) + # added to fix https://github.com/ModelOriented/DALEX/issues/400
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
  pl
}

plot.model_performance_histogram <- function(df, nlabels) {
  diff <- label <- NULL
  # commented to keep it consistent with other plots
  # see: https://github.com/ModelOriented/DALEX/issues/400
  # if (length(levels(df$label)) > 1) levels(df$label) <- rev(levels(df$label))

  ggplot(df, aes(diff, fill = label)) +
    geom_histogram(bins = 100) +
    facet_wrap(~label, ncol = 1) +
    theme_default_dalex() + xlab("residuals") + theme(legend.position = "none") +
    scale_fill_manual(name = "Model", values = colors_discrete_drwhy(nlabels)) +
    ggtitle("Histogram for residuals")

}

# precision-recall curve
plot.model_performance_prc <- function(df, nlabels) {
  dfl <- split(df, factor(df$label))
  prcdfl <- lapply(dfl, function(df) {
    pred_sorted <- df[order(df$predicted, decreasing = TRUE), ]

    # assuming that y = 0/1 where 1 is the positive
    recall <- cumsum(pred_sorted$observed)/sum(pred_sorted$observed)
    precis <- cumsum(pred_sorted$observed)/seq_along(pred_sorted$observed)
    data.frame(precis = precis, recall = recall, label = df$label[1])
  })
  prcdf <- do.call(rbind, prcdfl)

  precis <- recall <- label <- NULL
  ggplot(prcdf, aes(x = recall, y = precis, color = label)) +
    geom_line() +
    theme_default_dalex() +
    scale_color_manual(name = "Model", values = colors_discrete_drwhy(nlabels)) +
    scale_x_continuous("Recall", limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous("Precision", limits = c(0, 1), expand = c(0, 0)) +
    coord_fixed() +
    ggtitle("Precision Recall Curve")

}

# receiver operating characteristic
plot.model_performance_roc <- function(df, nlabels) {
  dfl <- split(df, factor(df$label))
  rocdfl <- lapply(dfl, function(df) {
    # assuming that y = 0/1 where 1 is the positive
    tpr_tmp <- tapply(df$observed, df$predicted, sum)
    tpr <- c(0,cumsum(rev(tpr_tmp)))/sum(df$observed)
    fpr_tmp <- tapply(1 - df$observed, df$predicted, sum)
    fpr <- c(0,cumsum(rev(fpr_tmp)))/sum(1 - df$observed)

    data.frame(tpr = tpr, fpr = fpr, label = df$label[1])
  })
  rocdf <- do.call(rbind, rocdfl)

  fpr <- tpr <- label <- NULL
  ggplot(rocdf, aes(x = fpr, y = tpr, color = label)) +
    geom_abline(slope = 1, intercept = 0, color = "grey", lty = 2) +
    geom_line() +
    theme_default_dalex() +
    scale_color_manual(name = "Model", values = colors_discrete_drwhy(nlabels)) +
    scale_x_continuous("False positive rate", limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous("True positive rate", limits = c(0, 1), expand = c(0, 0)) +
    coord_fixed() +
    ggtitle("Receiver Operator Characteristic")
}

plot.model_performance_gain <- function(df, nlabels) {
  dfl <- split(df, factor(df$label))
  rocdfl <- lapply(dfl, function(df) {
    pred_sorted <- df[order(df$predicted, decreasing = TRUE), ]

    # assuming that y = 0/1 where 1 is the positive
    lift <- cumsum(pred_sorted$observed)/length(pred_sorted$observed)
    pr <- seq_along(pred_sorted$observed)/length(pred_sorted$observed)
    data.frame(lift = lift, pr = pr, label = df$label[1])
  })
  rocdf <- do.call(rbind, rocdfl)
  max_lift <- sum(df$observed)/nrow(df)

  pr <- lift <- label <- NULL
  ggplot(rocdf, aes(x = pr, y = lift, color = label)) +
    geom_abline(slope = max_lift, intercept = 0, color = "grey", lty = 2) +
    geom_line() +
    theme_default_dalex() +
    scale_color_manual(name = "Model", values = colors_discrete_drwhy(nlabels)) +
    scale_x_continuous("Positive rate", limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous("True positive rate",  expand = c(0, 0)) +
    ggtitle("Cumulative Gains chart")

}


plot.model_performance_lift <- function(df, nlabels) {
  dfl <- split(df, factor(df$label))
  rocdfl <- lapply(dfl, function(df) {
    pred_sorted <- df[order(df$predicted, decreasing = TRUE), ]

    # assuming that y = 0/1 where 1 is the positive
    lift <- cumsum(pred_sorted$observed)/length(pred_sorted$observed)
    pr <- seq_along(pred_sorted$observed)/length(pred_sorted$observed)
    data.frame(lift = lift/pr, pr = pr, label = df$label[1])
  })
  rocdf <- do.call(rbind, rocdfl)
  max_lift <- sum(df$observed)/nrow(df)

  pr <- lift <- label <- NULL
  ggplot(rocdf, aes(x = pr, y = lift/max_lift, color = label)) +
    geom_abline(slope = 0, intercept = 1, color = "grey", lty = 2) +
    geom_line() +
    theme_default_dalex() +
    scale_color_manual(name = "Model", values = colors_discrete_drwhy(nlabels)) +
    scale_x_continuous("Positive rate", limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous("Lift",  expand = c(0, 0)) +
    ggtitle("Lift chart")

}
