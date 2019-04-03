.onAttach <- function(...) {
  addons <- setdiff(c("ALEPlot", "breakDown", "pdp", "factorMerger", "ggpubr"),
                    rownames(installed.packages()))

  packageStartupMessage("Welcome to DALEX (version: ", utils::packageVersion("DALEX"), ").\n",
    ifelse(length(addons) == 0, "",
    paste0( "Additional features will be available after installation of: ",paste(addons, collapse = ", "),".\nUse 'install_dependencies()' to get all suggested dependencies"))
        )
}
