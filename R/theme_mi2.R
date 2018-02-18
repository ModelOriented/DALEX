#' MI^2 Theme
#'
#' @return theme object that can be added to ggplot2 plots
#'
#' @export
#'
theme_mi2 <- function() {
  # serif instead of Tahoma to avoid problems with fonts
  # please fix some day
  theme(axis.ticks = element_line(linetype = "blank"),
        axis.title = element_text(family = "serif"),
        plot.title = element_text(family = "serif"),
        legend.text = element_text(family = "serif"),
        legend.title = element_text(family = "serif"),
        panel.background = element_rect(fill = "gray95"),
        plot.background = element_rect(fill = "gray95",
                                       colour = "aliceblue", size = 0.8,
                                       linetype = "dotted"), strip.background = element_rect(fill = "gray50"),
        strip.text = element_text(family = "serif"),
        legend.key = element_rect(fill = NA, colour = NA,
                                  size = 0),
        legend.background = element_rect(fill = NA))
}
