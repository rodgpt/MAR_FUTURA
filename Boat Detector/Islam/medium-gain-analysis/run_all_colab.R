if (!exists("project_root")) {
  project_root <- "/content/drive/Shareddrives/MAR FUTURA/CODES/Boat Detector/Islam/medium-gain-analysis"
}
if (!exists("drive_input_root")) {
  drive_input_root <- "/content/drive/Shareddrives/MAR FUTURA/Hydrophones/Matanzas/13-11-25/32"
}
if (!exists("selected_site")) {
  selected_site <- "Matanzas 32"
}
if (!exists("drive_output_root")) {
  drive_output_root <- "/content/drive/Shareddrives/MAR FUTURA/Hydrophones/BoatDetectionResults"
}
if (!exists("local_run_root")) {
  local_run_root <- "/content/boat_detection_run"
}
if (!exists("copy_outputs_to_drive")) {
  copy_outputs_to_drive <- TRUE
}

required_scripts <- c(
  "01_extract_features.R",
  "02_boat_detection.R",
  "03_report_events.R"
)

valid_sites <- c(
  "Las Cruces 26",
  "Matanzas 32",
  "San Antonio 38",
  "Ventanas 36",
  "Zapallar 34"
)

if (!dir.exists(project_root)) {
  stop(sprintf("project_root does not exist: %s", project_root))
}

if (!dir.exists(drive_input_root)) {
  stop(sprintf("drive_input_root does not exist: %s", drive_input_root))
}

if (!selected_site %in% valid_sites) {
  stop(
    sprintf(
      "selected_site must be one of: %s",
      paste(valid_sites, collapse = ", ")
    )
  )
}

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

if (copy_outputs_to_drive) {
  dir.create(drive_output_root, recursive = TRUE, showWarnings = FALSE)
}

dir.create(local_run_root, recursive = TRUE, showWarnings = FALSE)
local_project_root <- file.path(local_run_root, "project")
local_input_root <- file.path(local_run_root, "input", selected_site)

unlink(local_project_root, recursive = TRUE, force = TRUE)
unlink(file.path(local_run_root, "input"), recursive = TRUE, force = TRUE)

dir.create(local_project_root, recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(local_input_root), recursive = TRUE, showWarnings = FALSE)
dir.create(local_input_root, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(local_project_root, "outputs"), recursive = TRUE, showWarnings = FALSE)

for (script_name in required_scripts) {
  ok <- file.copy(
    file.path(project_root, script_name),
    file.path(local_project_root, script_name),
    overwrite = TRUE
  )
  if (!ok) {
    stop(sprintf("Failed to copy script to local project: %s", script_name))
  }
}

cat(sprintf("Project root on Drive: %s\n", project_root))
cat(sprintf("Drive input root: %s\n", drive_input_root))
cat(sprintf("Selected site: %s\n", selected_site))
cat(sprintf("Local run root: %s\n\n", local_run_root))
flush.console()

cat("Staging WAV files from Drive to local Colab disk...\n")
flush.console()

rsync_args <- c(
  "-a",
  "--info=progress2",
  "--prune-empty-dirs",
  "--include=*/",
  "--include=*.wav",
  "--include=*.WAV",
  "--exclude=*",
  paste0(normalizePath(drive_input_root, winslash = "/", mustWork = TRUE), "/"),
  paste0(normalizePath(local_input_root, winslash = "/", mustWork = TRUE), "/")
)

rsync_status <- system2("rsync", args = rsync_args)
if (!identical(rsync_status, 0L)) {
  stop(sprintf("rsync failed with status %s", rsync_status))
}

input_data_root <- local_input_root
wav_count <- length(
  list.files(
    input_data_root,
    pattern = "\\.wav$",
    ignore.case = TRUE,
    recursive = FALSE,
    full.names = TRUE
  )
)

cat(sprintf("Staged %d WAV files to local disk\n", wav_count))
if (wav_count == 0) {
  stop("No WAV files were staged to local disk. Check drive_input_root.")
}
cat("\n")
flush.console()

original_wd <- getwd()
on.exit(setwd(original_wd), add = TRUE)
setwd(local_project_root)

pipeline_start <- Sys.time()
cat(sprintf("Pipeline progress: 0/%d steps completed\n\n", length(required_scripts)))
flush.console()

for (i in seq_along(required_scripts)) {
  script_name <- required_scripts[i]
  cat(sprintf("============================================================\n"))
  cat(sprintf("[%d/%d] Running %s\n", i, length(required_scripts), script_name))
  cat(sprintf("============================================================\n"))
  flush.console()

  step_start <- Sys.time()
  source(file.path(local_project_root, script_name), local = FALSE)
  step_elapsed <- difftime(Sys.time(), step_start, units = "secs")

  cat(sprintf("Completed step %d in %.1f seconds\n", i, as.numeric(step_elapsed)))
  cat(sprintf("Pipeline progress: %d/%d steps completed\n\n", i, length(required_scripts)))
  flush.console()
}

pipeline_elapsed <- difftime(Sys.time(), pipeline_start, units = "secs")

feature_file <- file.path(local_project_root, "boat_features_all.csv")
detections_file <- file.path(local_project_root, "outputs", "boat_detections_FINAL.csv")
events_file <- file.path(local_project_root, "outputs", "vessel_events_FINAL.csv")

if (!file.exists(feature_file)) {
  stop("Missing expected output: boat_features_all.csv")
}
if (!file.exists(detections_file)) {
  stop("Missing expected output: outputs/boat_detections_FINAL.csv")
}
if (!file.exists(events_file)) {
  stop("Missing expected output: outputs/vessel_events_FINAL.csv")
}

if (copy_outputs_to_drive) {
  drive_site_output <- file.path(drive_output_root, gsub("/", "_", selected_site))
  dir.create(drive_site_output, recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(drive_site_output, "outputs"), recursive = TRUE, showWarnings = FALSE)

  file.copy(feature_file, file.path(drive_site_output, "boat_features_all.csv"), overwrite = TRUE)
  file.copy(detections_file, file.path(drive_site_output, "outputs", "boat_detections_FINAL.csv"), overwrite = TRUE)
  file.copy(events_file, file.path(drive_site_output, "outputs", "vessel_events_FINAL.csv"), overwrite = TRUE)

  cat(sprintf("Copied outputs back to Drive: %s\n", drive_site_output))
}

cat("Pipeline completed successfully.\n")
cat(sprintf("Total elapsed time: %.1f seconds\n", as.numeric(pipeline_elapsed)))
cat(sprintf("Local features: %s\n", feature_file))
cat(sprintf("Local detections: %s\n", detections_file))
cat(sprintf("Local events: %s\n", events_file))
flush.console()
