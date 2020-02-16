#' Install all dependencies for the DALEX package
#'
#' By default 'heavy' dependencies are not installed along DALEX.
#' This function silently install all required packages.
#'
#' @param packages which packages shall be installed?
#' @importFrom utils install.packages
#' @export
install_dependencies <- function(packages = c("ingredients", "iBreakDown", "ggpubr")) {
  sapply(packages, function(package) {
    cat("\nInstallation of the ", package, " initiated.\n")
    try(install.packages(package), silent = TRUE)
  })
  invisible(NULL)
}
