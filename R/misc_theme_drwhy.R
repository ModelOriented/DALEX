#' DrWhy Theme for ggplot objects
#'
#' @return theme for ggplot2 objects
#' @export
#' @rdname theme_drwhy
theme_drwhy <- function() {
    theme_bw(base_line_size = 1) %+replace%
    theme(axis.ticks = element_blank(), legend.background = element_blank(),
          legend.key = element_blank(), panel.background = element_blank(),
          panel.border = element_blank(), strip.background = element_blank(),
          plot.background = element_blank(), complete = TRUE,
          legend.direction = "horizontal", legend.position = "top",
          axis.line.y = element_line(color = "white"),
          axis.ticks.y = element_line(color = "white"),
          #axis.line = element_line(color = "#371ea3", linewidth = 0.5, linetype = 1),
          axis.title = element_text(color = "#371ea3"),
          plot.title = element_text(color = "#371ea3", size = 16, hjust = 0),
          plot.subtitle = element_text(color = "#371ea3", hjust = 0),
          axis.text = element_text(color = "#371ea3", size = 10),
          strip.text = element_text(color = "#371ea3", size = 12, hjust = 0),
          panel.grid.major.y = element_line(color = "grey90", linewidth = 0.5, linetype = 1),
          panel.grid.minor.y = element_line(color = "grey90", linewidth = 0.5,  linetype = 1),
          panel.grid.minor.x = element_blank(),
          panel.grid.major.x = element_blank())

}

#' @export
#' @rdname theme_drwhy
theme_ema <- function() {
  theme_drwhy() %+replace%
    theme(text = element_text(color = "black", size = 12),
          plot.title = element_text(color = "black", size = 14, hjust = 0),
          plot.subtitle = element_blank(),
          axis.text = element_text(color = "black", size = 12),
          axis.text.x = element_text(color = "black", size = 12),
          axis.text.y = element_text(color = "black", size = 12),
          axis.title = element_text(color = "black", size = 12),
          legend.text = element_text(color = "black", size = 12),
          legend.position = "none",
          strip.text = element_text(color = "black", size = 12, hjust = 0))
}

#' @export
#' @rdname theme_drwhy
theme_drwhy_vertical <- function() {
  theme_bw(base_line_size = 1) %+replace%
    theme(axis.ticks = element_blank(), legend.background = element_blank(),
          legend.key = element_blank(), panel.background = element_blank(),
          panel.border = element_blank(), strip.background = element_blank(),
          plot.background = element_blank(), complete = TRUE,
          legend.direction = "horizontal", legend.position = "top",
          axis.line.x = element_line(color = "white"),
          axis.ticks.x = element_line(color = "white"),
          plot.title = element_text(color = "#371ea3", size = 16, hjust = 0),
          plot.subtitle = element_text(color = "#371ea3", hjust = 0),
          #axis.line = element_line(color = "#371ea3", linewidth = 0.5, linetype = 1),
          axis.title = element_text(color = "#371ea3"),
          axis.text = element_text(color = "#371ea3", size = 10),
          strip.text = element_text(color = "#371ea3", size = 12, hjust = 0),
          panel.grid.major.x = element_line(color = "grey90", linewidth = 0.5, linetype = 1),
          panel.grid.minor.x = element_line(color = "grey90", linewidth = 0.5,  linetype = 1),
          panel.grid.minor.y = element_blank(),
          panel.grid.major.y = element_blank())
}

#' @export
#' @rdname theme_drwhy
theme_ema_vertical <- function() {
  theme_drwhy_vertical() %+replace%
    theme(text = element_text(color = "black", size = 12),
          plot.title = element_text(color = "black", size = 14, hjust = 0),
          plot.subtitle = element_blank(),
          axis.text = element_text(color = "black", size = 12),
          axis.text.x = element_text(color = "black", size = 12),
          axis.text.y = element_text(color = "black", size = 12),
          axis.title = element_text(color = "black", size = 12),
          legend.text = element_text(color = "black", size = 12),
          legend.position = "none",
          strip.text = element_text(color = "black", size = 12, hjust = 0))
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
