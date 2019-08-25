#' DrWhy Theme for ggplot objects
#'
#' @return theme for ggplot2 objects
#' @export
#' @rdname theme_drwhy
theme_drwhy <- function() {
    theme_bw(base_line_size = 0) %+replace%
    theme(axis.ticks = element_blank(), legend.background = element_blank(),
          legend.key = element_blank(), panel.background = element_blank(),
          panel.border = element_blank(), strip.background = element_blank(),
          plot.background = element_blank(), complete = TRUE,
          legend.direction = "horizontal", legend.position = "top",
          axis.line.y = element_line(color = "white"),
          axis.ticks.y = element_line(color = "white"),
          #axis.line = element_line(color = "#371ea3", size = 0.5, linetype = 1),
          axis.title = element_text(color = "#371ea3"),
          plot.title = element_text(color = "#371ea3", size = 16),
          axis.text = element_text(color = "#371ea3", size = 10),
          strip.text = element_text(color = "#371ea3", size = 12, hjust = 0),
          panel.grid.major.y = element_line(color = "grey90", size = 0.5, linetype = 1),
          panel.grid.minor.y = element_line(color = "grey90", size = 0.5,  linetype = 1))

}

#' @export
#' @rdname theme_drwhy
theme_drwhy_vertical <- function() {
  theme_bw(base_line_size = 0) %+replace%
    theme(axis.ticks = element_blank(), legend.background = element_blank(),
          legend.key = element_blank(), panel.background = element_blank(),
          panel.border = element_blank(), strip.background = element_blank(),
          plot.background = element_blank(), complete = TRUE,
          legend.direction = "horizontal", legend.position = "top",
          axis.line.x = element_line(color = "white"),
          axis.ticks.x = element_line(color = "white"),
          plot.title = element_text(color = "#371ea3", size = 16),
          #axis.line = element_line(color = "#371ea3", size = 0.5, linetype = 1),
          axis.title = element_text(color = "#371ea3"),
          axis.text = element_text(color = "#371ea3", size = 10),
          strip.text = element_text(color = "#371ea3", size = 12, hjust = 0),
          panel.grid.major.x = element_line(color = "grey90", size = 0.5, linetype = 1),
          panel.grid.minor.x = element_line(color = "grey90", size = 0.5,  linetype = 1))

}


#' DrWhy color palettes for ggplot objects
#'
#' @param n number of colors for color palette
#'
#' @return color palette as vector of charactes
#' @export
#' @rdname colors_drwhy
colors_discrete_drwhy <- function(n = 2) {
  if (n == 1) return("#4378bf")
  if (n == 2) return(c( "#4378bf", "#8bdcbe"))
  if (n == 3) return(c( "#4378bf", "#f05a71", "#8bdcbe"))
  if (n == 4) return(c( "#4378bf", "#f05a71", "#8bdcbe", "#ffa58c"))
  if (n == 5) return(c( "#4378bf", "#f05a71", "#8bdcbe", "#ae2c87", "#ffa58c"))
  if (n == 6) return(c( "#4378bf", "#46bac2", "#8bdcbe", "#ae2c87", "#ffa58c", "#f05a71"))
  c( "#4378bf", "#46bac2", "#371ea3", "#8bdcbe", "#ae2c87", "#ffa58c", "#f05a71")[((0:(n-1)) %% 7) + 1]
}


#' @export
#' @rdname colors_drwhy
colors_diverging_drwhy <- function() {
  c("#c7f5bf", "#371ea3")
}


#' @export
#' @rdname colors_drwhy
colors_breakdown_drwhy <- function() {
  c(`-1` = "#f05a71", `0` = "#371ea3", `1` = "#8bdcbe", X = "#371ea3")
}

#' @export
#' @rdname colors_drwhy
theme_drwhy_colors  <- function(n = 2) {
  # Deprecated, but print the message only once
  if (!exists("message_theme_colors", envir = .DALEX.env)) {
    .DALEX.env$message_theme_colors = TRUE
    .Deprecated("theme_drwhy_colors()",  msg = "Please note that 'theme_drwhy_colors()' is now deprecated, it is better to use 'colors_discrete_drwhy()' instead.")
  }
  colors_discrete_drwhy(n)
}

#' @export
#' @rdname colors_drwhy
theme_drwhy_colors_break_down <- function() {
  # Deprecated, but print the message only once
  if (!exists("message_theme_colors_breakdown", envir = .DALEX.env)) {
    .DALEX.env$message_theme_colors_breakdown = TRUE
    .Deprecated("theme_drwhy_colors_break_down()",  msg = "Please note that 'theme_drwhy_colors_break_down()' is now deprecated, it is better to use 'colors_breakdown_drwhy' instead.")
  }
  colors_breakdown_drwhy()
}


#' MI^2 Theme
#'
#' @export
#' @rdname theme_drwhy
theme_mi2 <- function() {
  # sans instead of Tahoma to avoid problems with fonts
  theme(axis.ticks = element_line(linetype = "blank"),
        axis.title = element_text(family = "sans"),
        plot.title = element_text(family = "sans"),
        legend.text = element_text(family = "sans"),
        legend.title = element_text(family = "sans"),
        panel.background = element_rect(fill = "#f5f5f5"),
        plot.background = element_rect(fill = "#f5f5f5",
                                       colour = "aliceblue", size = 0.8,
                                       linetype = "dotted"), strip.background = element_rect(fill = "gray50"),
        strip.text = element_text(family = "sans"),
        legend.key = element_rect(fill = NA, colour = NA,
                                  size = 0),
        legend.background = element_rect(fill = NA))
}
