.onAttach <- function(...) {
  packageStartupMessage("Welcome to DALEX (version: ", utils::packageVersion("DALEX"), ").\n",
        "This is a plain DALEX. Use 'install_dependencies()' to get all required packages.")
}
