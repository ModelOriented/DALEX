#' Instance Level Residual Diagnostics
#'
#' This function performs local diagnostic of residuals.
#' For a single instance its neighbors are identified in the validation data.
#' Residuals are calculated for neighbors and plotted against residuals for all data.
#' Find information how to use this function here: \url{https://pbiecek.github.io/ema/localDiagnostics.html}.
#'
#' @param explainer a model to be explained, preprocessed by the 'explain' function
#' @param new_observation a new observation for which predictions need to be explained
#' @param variables character - name of variables to be explained
#' @param ... other parameters
#' @param nbins number of bins for the histogram. By default 20
#' @param neighbors number of neighbors for histogram. By default 50.
#' @param distance the distance function, by default the \code{gower_dist()} function.
#'
#' @return An object of the class 'predict_diagnostics'.
#' It's a data frame with calculated distribution of residuals.
#'
#' @references Explanatory Model Analysis. Explore, Explain and Examine Predictive Models. \url{https://pbiecek.github.io/ema/}
#' @export
#' @importFrom stats ks.test
#' @importFrom graphics plot
#' @examples
#' library("ranger")
#' titanic_glm_model <- ranger(survived ~ gender + age + class + fare + sibsp + parch,
#'                      data = titanic_imputed)
#' explainer_glm <- explain(titanic_glm_model,
#'                          data = titanic_imputed,
#'                          y = titanic_imputed$survived)
#' johny_d <- titanic_imputed[24, c("gender", "age", "class", "fare", "sibsp", "parch")]
#'
#' \dontrun{
#' id_johny <- predict_diagnostics(explainer_glm, johny_d, variables = NULL)
#' id_johny
#' plot(id_johny)
#'
#' id_johny <- predict_diagnostics(explainer_glm, johny_d,
#'                        neighbors = 10,
#'                        variables = c("age", "fare"))
#' id_johny
#' plot(id_johny)
#'
#' id_johny <- predict_diagnostics(explainer_glm,
#'                        johny_d,
#'                        neighbors = 10,
#'                        variables = c("class", "gender"))
#' id_johny
#' plot(id_johny)
#'}
#'
#' @name predict_diagnostics
#' @export
predict_diagnostics <-  function(explainer, new_observation, variables = NULL, ..., nbins = 20, neighbors = 50, distance = gower::gower_dist) {
  test_expaliner(explainer, has_data = TRUE, function_name = "predict_diagnostics")

  neighbours_id <- select_neighbours_id(new_observation, explainer$data, n = neighbors, distance = distance)


  # if variables = NULL then histograms with distribution of residuals are compared against each other
  if (is.null(variables)) {
    residuals_all <- explainer$residual_function(explainer$model, explainer$data, explainer$y)
    residuals_sel <- residuals_all[neighbours_id]

    cut_points <- signif(pretty(residuals_all, nbins), 3)
    test.res <- ks.test(residuals_all, residuals_sel)

    df1 <- data.frame(as.data.frame(table(cut(residuals_sel, cut_points))/length(residuals_sel)), direction = "neighbours")
    df2 <- data.frame(as.data.frame(-table(cut(residuals_all, cut_points))/length(residuals_all)), direction = "all")

    res <- list(variables = variables,
                histogram_neighbours = df1,
                histogram_all = df2,
                test = test.res,
                cut_points = cut_points,
                neighbours_id = neighbours_id)
  } else {
    # if variables is not null then we need to plot either categorical or continouse fidelity plot
    cp_neighbors <- ingredients::ceteris_paribus(explainer,
                                                 new_observation = explainer$data[neighbours_id, ],
                                                 y = explainer$y[neighbours_id],
                                                 variables = variables,
                                                 ...)
    cp_new_instance <- ingredients::ceteris_paribus(explainer,
                                                 new_observation = new_observation,
                                                 variables = variables,
                                                 ...)
    res <- list(variables = variables,
                cp_neighbors = cp_neighbors,
                cp_new_instance = cp_new_instance,
                neighbours_id = neighbours_id)
  }
  class(res) <- "predict_diagnostics"
  res
}

#' @name predict_diagnostics
#' @export
individual_diagnostics <- predict_diagnostics


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
