#' Default Theme for DALEX plots
#'
#' @param default_theme object - string ("drwhy" or "ema") or an object of ggplot theme class. Will be applied by default by DALEX to all horizontal plots
#' @param default_theme_vertical object - string ("drwhy" or "ema") or an object of ggplot theme class. Will be applied by default by DALEX to all vertical plots
#' @return list with current default themes
#' @examples
#' old <- set_theme_dalex("ema")
#' \donttest{
#' library("ranger")
#' apartments_ranger_model <- ranger(m2.price~., data = apartments, num.trees = 50)
#' explainer_ranger  <- explain(apartments_ranger_model, data = apartments[,-1],
#'                              y = apartments$m2.price, label = "Ranger Apartments")
#' model_parts_ranger_aps <- model_parts(explainer_ranger, type = "raw")
#' head(model_parts_ranger_aps, 8)
#' plot(model_parts_ranger_aps)
#'
#' old <- set_theme_dalex(ggplot2::theme_void(), ggplot2::theme_void())
#' plot(model_parts_ranger_aps)
#'
#' old <- set_theme_dalex("drwhy")
#' plot(model_parts_ranger_aps)
#' old <- set_theme_dalex(ggplot2::theme_void(), ggplot2::theme_void())
#' plot(model_parts_ranger_aps)
#'}
#'
#' @export
#' @rdname theme_dalex
#'
set_theme_dalex <- function(default_theme = "drwhy", default_theme_vertical = default_theme) {
  # it should be either name or theme object
  if (!(any(
    class(default_theme) %in% c("character", "theme")
    )))
      stop("The 'default_theme' shall be either character 'drwhy'/'ema' or ggplot2::theme object")
  if (!(any(
    class(default_theme_vertical) %in% c("character", "theme")
    )))
      stop("The 'default_theme_vertical' shall be either character 'drwhy'/'ema' or ggplot2::theme object")

  # get default themes
  old <- .DALEX.env$default_themes

  # set themes
  if (is.character(default_theme)) {
    # from name
    switch (default_theme,
      drwhy = {.DALEX.env$default_themes <- list(default = theme_drwhy(), vertical = theme_drwhy_vertical())},
      ema = {.DALEX.env$default_themes <- list(default = theme_ema(), vertical = theme_ema_vertical())},
      stop("Only 'drwhy' or 'ema' names are allowed")
    )
  } else {
    # from themes (ggplot2 objects)
    .DALEX.env$default_themes <- list(default = default_theme, vertical = default_theme_vertical)
  }

  # return old themes
  old
}


#' @export
#' @rdname theme_dalex
theme_default_dalex <- function() {
  if (!exists("default_themes", envir = .DALEX.env))
      return(theme_drwhy())

  .DALEX.env$default_themes[[1]]
}

#' @export
#' @rdname theme_dalex
theme_vertical_default_dalex <- function() {
  if (!exists("default_themes", envir = .DALEX.env))
    return(theme_drwhy_vertical())

  .DALEX.env$default_themes[[2]]
}

