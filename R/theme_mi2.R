#' MI^2 Theme
#'
#' @return theme object that can be added to ggplot2 plots
#'
#' @export
#'
theme_mi2 <- function() {
  # sans instead of Tahoma to avoid problems with fonts
  # please fix some day
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
