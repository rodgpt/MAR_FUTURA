project_root <- "/Users/rodrigo/Desktop/CODES/Boat Detector/Islam/medium-gain-analysis"
input_data_root <- "/Users/rodrigo/Library/CloudStorage/GoogleDrive-royanedel@marfutura.org/Unidades compartidas/Hydrophones/Matanzas/13-11-25/44"
selected_site <- "Matanzas 32"

if (!dir.exists(project_root)) {
  stop(sprintf("project_root does not exist: %s", project_root))
}

if (!dir.exists(input_data_root)) {
  stop(sprintf("input_data_root does not exist: %s", input_data_root))
}

valid_sites <- c(
  "Las Cruces 26",
  "Matanzas 32",
  "San Antonio 38",
  "Ventanas 36",
  "Zapallar 34"
)

if (!selected_site %in% valid_sites) {
  stop(
    sprintf(
      "selected_site must be one of: %s",
      paste(valid_sites, collapse = ", ")
    )
  )
}

cat(sprintf("Running boat detection pipeline in: %s\n", project_root))
cat(sprintf("Reading WAV files from: %s\n", input_data_root))
cat(sprintf("Using site label: %s\n", selected_site))
cat("Scanning input folder for WAV files...\n")
flush.console()

wav_count <- length(
  list.files(
    input_data_root,
    pattern = "\\.wav$",
    ignore.case = TRUE,
    recursive = FALSE,
    full.names = TRUE
  )
)

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

step1_root <- tempfile(pattern = "boat_detection_input_")
dir.create(step1_root, recursive = TRUE)
on.exit(unlink(step1_root, recursive = TRUE, force = TRUE), add = TRUE)

step1_site_dir <- file.path(step1_root, selected_site)
linked <- file.symlink(input_data_root, step1_site_dir)
if (!isTRUE(linked)) {
  dir.create(step1_site_dir, recursive = TRUE)
  wav_files <- list.files(input_data_root, full.names = TRUE, all.files = FALSE, no.. = TRUE)
  copied <- file.copy(wav_files, step1_site_dir, recursive = TRUE)
  if (length(wav_files) > 0 && !all(copied)) {
    stop("Failed to prepare temporary input folder for 01_extract_features.R")
  }
}

pipeline_start <- Sys.time()

cat(sprintf("============================================================\n"))
cat(sprintf("[1/%d] Running %s\n", length(required_scripts), required_scripts[1]))
cat(sprintf("============================================================\n"))
flush.console()
step_start <- Sys.time()
setwd(step1_root)
source(file.path(project_root, required_scripts[1]), local = FALSE)
step_elapsed <- difftime(Sys.time(), step_start, units = "secs")
cat(sprintf("Completed step 1 in %.1f seconds\n", as.numeric(step_elapsed)))
cat(sprintf("Pipeline progress: 1/%d steps completed\n", length(required_scripts)))
cat("\n")
flush.console()

step1_output <- file.path(step1_root, "boat_features_all.csv")
if (!file.exists(step1_output)) {
  stop("01_extract_features.R did not create boat_features_all.csv")
}

file.copy(step1_output, file.path(project_root, "boat_features_all.csv"), overwrite = TRUE)

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
