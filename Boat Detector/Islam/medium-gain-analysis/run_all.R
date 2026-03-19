project_root <- "/Users/rodrigo/Desktop/CODES/Boat Detector/Islam/medium-gain-analysis"
run_all_sites <- TRUE
selected_site <- "Matanzas 32"

site_input_paths <- c(
  "Las Cruces 26" = "/Volumes/PortableSSD/Hydrophones/LasCruces/12-11-25/26",
  "Las Cruces 41" = "/Volumes/PortableSSD/Hydrophones/LasCruces/20-10-25/41",
  "Matanzas 32" = "/Volumes/PortableSSD/Hydrophones/Matanzas/13-11-25/32",
  "Ventanas 36" = "/Volumes/PortableSSD/Hydrophones/Ventanas/07-11-25/36",
  "Ventanas 38" = "/Volumes/PortableSSD/Hydrophones/Ventanas/20-10-25/38",
  "Zapallar 34" = "/Volumes/PortableSSD/Hydrophones/Zapallar/07-11-25/34",
  "Zapallar 32" = "/Volumes/PortableSSD/Hydrophones/Zapallar/20-10-25/32"
)

valid_sites <- names(site_input_paths)

if (!dir.exists(project_root)) {
  stop(sprintf("project_root does not exist: %s", project_root))
}

if (isTRUE(run_all_sites)) {
  selected_site <- names(site_input_paths)[1]
} else {
  if (!selected_site %in% valid_sites) {
    stop(
      sprintf(
        "selected_site must be one of: %s",
        paste(valid_sites, collapse = ", ")
      )
    )
  }
}

input_data_root <- site_input_paths[[selected_site]]

if (is.null(input_data_root) || is.na(input_data_root) || input_data_root == "") {
  stop(sprintf("No input path configured for selected_site: %s", selected_site))
}

if (!dir.exists(input_data_root)) {
  stop(sprintf("input_data_root does not exist: %s", input_data_root))
}

cat(sprintf("Running boat detection pipeline in: %s\n", project_root))
if (isTRUE(run_all_sites)) {
  cat(sprintf("Reading WAV files from %d site folders\n", length(site_input_paths)))
} else {
  cat(sprintf("Reading WAV files from: %s\n", input_data_root))
  cat(sprintf("Using site label: %s\n", selected_site))
}
cat("Scanning input folder for WAV files...\n")
flush.console()

wav_count <- if (isTRUE(run_all_sites)) {
  sum(
    vapply(
      site_input_paths,
      function(p) length(list.files(p, pattern = "\\.wav$", ignore.case = TRUE, recursive = FALSE)),
      integer(1)
    )
  )
} else {
  length(
    list.files(
      input_data_root,
      pattern = "\\.wav$",
      ignore.case = TRUE,
      recursive = FALSE,
      full.names = TRUE
    )
  )
}

required_scripts <- c(
  "01_extract_features.R",
  "02_boat_detection.R",
  "03_report_events.R"
)

missing_scripts <- required_scripts[
  !file.exists(file.path(project_root, required_scripts))
]

if (length(missing_scripts) > 0) {
  stop(
    sprintf(
      "Missing required scripts in project_root: %s",
      paste(missing_scripts, collapse = ", ")
    )
  )
}

original_wd <- getwd()
on.exit(setwd(original_wd), add = TRUE)

cat(sprintf("Found %d WAV files in input_data_root\n", wav_count))
cat(sprintf("Pipeline progress: 0/%d steps completed\n\n", length(required_scripts)))
flush.console()

pipeline_start <- Sys.time()

cat(sprintf("============================================================\n"))
cat(sprintf("[1/%d] Running %s\n", length(required_scripts), required_scripts[1]))
cat(sprintf("============================================================\n"))
flush.console()
step_start <- Sys.time()
setwd(project_root)
if (isTRUE(run_all_sites)) {
  all_site_features <- list()
  for (site_name in names(site_input_paths)) {
    selected_site <- site_name
    input_data_root <- site_input_paths[[selected_site]]
    source(file.path(project_root, required_scripts[1]), local = FALSE)
    site_features <- read.csv(file.path(project_root, "boat_features_all.csv"), stringsAsFactors = FALSE)
    all_site_features[[length(all_site_features) + 1]] <- site_features
  }
  combined_features <- do.call(rbind, all_site_features)
  write.csv(combined_features, file.path(project_root, "boat_features_all.csv"), row.names = FALSE)
} else {
  source(file.path(project_root, required_scripts[1]), local = FALSE)
}
step_elapsed <- difftime(Sys.time(), step_start, units = "secs")
cat(sprintf("Completed step 1 in %.1f seconds\n", as.numeric(step_elapsed)))
cat(sprintf("Pipeline progress: 1/%d steps completed\n", length(required_scripts)))
cat("\n")
flush.console()

step1_output <- file.path(project_root, "boat_features_all.csv")
if (!file.exists(step1_output)) {
  stop("01_extract_features.R did not create boat_features_all.csv")
}

for (i in seq_along(required_scripts[-1])) {
  script_name <- required_scripts[-1][i]
  cat(sprintf("============================================================\n"))
  cat(sprintf("[%d/%d] Running %s\n", i + 1, length(required_scripts), script_name))
  cat(sprintf("============================================================\n"))
  flush.console()
  step_start <- Sys.time()
  setwd(project_root)
  source(file.path(project_root, script_name), local = FALSE)
  step_elapsed <- difftime(Sys.time(), step_start, units = "secs")
  cat(sprintf("Completed step %d in %.1f seconds\n", i + 1, as.numeric(step_elapsed)))
  cat(sprintf("Pipeline progress: %d/%d steps completed\n", i + 1, length(required_scripts)))
  cat("\n")
  flush.console()
}

pipeline_elapsed <- difftime(Sys.time(), pipeline_start, units = "secs")
cat("Pipeline completed successfully.\n")
cat(sprintf("Total elapsed time: %.1f seconds\n", as.numeric(pipeline_elapsed)))
cat(sprintf("Features: %s\n", file.path(project_root, "boat_features_all.csv")))
cat(sprintf("Detections: %s\n", file.path(project_root, "outputs", "boat_detections_FINAL.csv")))
cat(sprintf("Events: %s\n", file.path(project_root, "outputs", "vessel_events_FINAL.csv")))
flush.console()
