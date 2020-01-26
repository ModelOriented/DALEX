#' Instance Level Residuals Distribution
#'
#' This function performs local diagnostic of residuals.
#' For a single instance its neighbours are identified in the validation data.
#' Residuals are calculated for neighbours and plotted agains residuals for all data.
#' Find information how to use this function here: \url{https://pbiecek.github.io/ema/localDiagnostics.html}.
#'
#' @param explainer a model to be explained, preprocessed by the 'explain' function
#' @param new_observation a new observarvation for which predictions need to be explained
#' @param variables character - name of variables to be explained
#' @param ... other parameters
#' @param nbins number of bins for the histogram. By default 20
#' @param neighbours number of neighbours for histogram. By default 50.
#' @param distance the distance function, by default the \code{gower_dist()} function.
#'
#'
#' @return An object of the class 'residuals_distribution_explainer'.
#' It's a data frame with calculated distribution of residuals.
#'
#' @references Explanatory Model Analysis. Explore, Explain and Examine Predictive Models. \url{https://pbiecek.github.io/ema/}
#' @export
#' @importFrom gower gower_dist
#' @importFrom stats ks.test
#' @examples
#' titanic_glm_model <- glm(survived~., data = titanic_imputed, family = "binomial")
#' explainer_glm <- explain(titanic_glm_model, data = titanic_imputed)
#'

#' @name residuals_distribution
#' @export
residuals_distribution <-  function(explainer, new_observation, variables = NULL, ..., nbins = 20, neighbours = 50, distance = gower::gower_dist) {
  if (nrow(new_observation) != 1) stop("The residuals_distribution() function 'new_observation' with a single row.")
  if (is.null(explainer$data)) stop("The explainer needs to have the 'data' slot.")

  neighbours_id <- select_neighbours_id(new_observation, explainer$data, n = neighbours, distance = distance)


  # if variables = NULL then histograms with distribution of residuals are compared against each other
  if (is.null(variables)) {
    residuals_all <- explainer$residual_function(explainer$model, explainer$data, explainer$y)
    residuals_sel <- residuals_all[neighbours_id]

    cut_points <- signif(pretty(residuals_all, nbins), 3)
    test.res <- ks.test(residuals_all, residuals_sel)

    df1 <- data.frame(as.data.frame(table(cut(residuals_sel, cut_points))/length(residuals_sel)), direction = "neighbours")
    df2 <- data.frame(as.data.frame(-table(cut(residuals_all, cut_points))/length(residuals_all)), direction = "all")
    df <- rbind(df1, df2)

    pl <- ggplot(df, aes(Var1, Freq, fill = direction)) +
      geom_col() +
      scale_y_continuous("") +
      scale_x_discrete("residuals", labels = as.character(cut_points)) +
      scale_fill_manual("", values = colors_diverging_drwhy()) +
      theme_drwhy() + theme(legend.position = "top") +
      ggtitle("Distribution of residuals",
              paste0("Difference between distributions: D ", signif(test.res$statistic, 3),
                     " p.value ", signif(test.res$p.value, 3)))
  } else {

  }
  pl
}


select_neighbours_id <- function(observation, data, variables = NULL, distance = gower::gower_dist, n = 50, frac = NULL) {
  if (is.null(variables)) {
    variables <- intersect(colnames(observation),
                           colnames(data))
  }
  if (is.null(n)) {
    n <- ceiling(nrow(data)*frac)
  }

  distances <- distance(observation[,variables, drop = FALSE],
                        data[,variables, drop = FALSE])
  head(order(distances), n)
}
