library(tuneR)
library(seewave)
library(dplyr)
library(purrr)
library(ggplot2)
library(scales)
library(lubridate)
library(tidyr)
library(soundecology)
library(beepr)

site_label <- "San Antonio 28"
site_dir <- "/Users/rod/Library/CloudStorage/GoogleDrive-royanedel@marfutura.org/Unidades compartidas/MAR FUTURA/Hydrophones/San Antonio/18-10-25/28"

segment_sec <- 250
anthro_band <- c(1000, 2000)
bio_band    <- c(2000, 3000)

tz <- "UTC"
analysis_duration <- NA
files_per_folder <- NA

start_date <- NA
end_date   <- NA

extract_datetime <- function(filename) {
  dt_str <- sub("^(?:ST_\\d+_)?(\\d{8}_\\d{6})\\.WAV$", "\\1", basename(filename), ignore.case = TRUE)
  as.POSIXct(dt_str, format = "%Y%m%d_%H%M%S", tz = tz)
}

calculate_ndsi <- function(wave_obj) {
  nd <- ndsi(
    wave_obj,
    anthro_min = anthro_band[1], anthro_max = anthro_band[2],
    bio_min    = bio_band[1],    bio_max    = bio_band[2]
  )

  samples <- wave_obj@left
  sample_rate <- wave_obj@samp.rate
  n <- length(samples)

  fft_result <- fft(samples)
  power_spectrum <- Mod(fft_result[1:(n/2)])^2 / n^2
  freqs_hz <- (0:(n/2 - 1)) * (sample_rate / as.double(n))

  anthro_indices <- which(freqs_hz >= anthro_band[1] & freqs_hz <= anthro_band[2])
  anthro_energy <- sum(power_spectrum[anthro_indices])

  bio_indices <- which(freqs_hz >= bio_band[1] & freqs_hz <= bio_band[2])
  bio_energy <- sum(power_spectrum[bio_indices])

  list(ndsi = nd$ndsi_left, anthro_energy = anthro_energy, bio_energy = bio_energy)
}

process_site <- function(directory, label) {
  files <- list.files(directory, pattern = "\\.wav$", full.names = TRUE, recursive = TRUE, ignore.case = TRUE)
  message("Found ", length(files), " files in ", label, " (searching recursively, case-insensitive)")

  if (length(files) == 0) {
    return(tibble(
      Site = character(),
      Time = as.POSIXct(character()),
      NDSI = numeric(),
      Anthro_Energy = numeric(),
      Bio_Energy = numeric()
    ))
  }

  files <- sort(files)

  file_dt <- purrr::map_df(files, ~tibble(
    filepath = .x,
    start_dt = extract_datetime(.x)
  ))

  if (!is.na(start_date)) {
    file_dt <- dplyr::filter(file_dt, start_dt >= (start_date - segment_sec))
  }
  if (!is.na(end_date)) {
    file_dt <- dplyr::filter(file_dt, start_dt <= end_date)
  }

  file_dt <- dplyr::arrange(file_dt, start_dt)
  files <- file_dt$filepath

  if (!is.na(files_per_folder)) {
    files <- head(files, files_per_folder)
  }

  results <- list()

  for (fp in files) {
    message("Reading: ", fp)
    start_dt <- extract_datetime(fp)

    wav <- tryCatch(readWave(fp), error = function(e) {
      warning("Skipping unreadable file: ", fp)
      return(NULL)
    })
    if (is.null(wav)) next

    dur_sec <- length(wav@left) / wav@samp.rate
    max_start <- max(0, dur_sec - segment_sec)
    starts <- seq(0, max_start, by = segment_sec)

    for (st in starts) {
      segment_time <- start_dt + st

      if (!is.na(start_date) && segment_time < start_date) {
        next
      }
      if (!is.na(end_date) && segment_time > end_date) {
        break
      }

      seg <- tryCatch(
        extractWave(wav, from = st, to = st + segment_sec, xunit = "time"),
        error = function(e) return(NULL)
      )
      if (is.null(seg)) next

      ndsi_res <- calculate_ndsi(seg)

      results[[length(results) + 1]] <- tibble(
        Site = label,
        Time = segment_time,
        NDSI = ndsi_res$ndsi,
        Anthro_Energy = ndsi_res$anthro_energy,
        Bio_Energy = ndsi_res$bio_energy
      )
    }
  }

  if (length(results) == 0) {
    return(tibble(
      Site = character(),
      Time = as.POSIXct(character()),
      NDSI = numeric(),
      Anthro_Energy = numeric(),
      Bio_Energy = numeric()
    ))
  }

  bind_rows(results)
}

all_results <- process_site(site_dir, site_label)

if (!exists("all_results") || nrow(all_results) == 0) {
  stop("No WAV files found in the provided directory and date range. Please verify `site_dir` and the date filters.")
}

anthro_tag <- paste0("anthro_", anthro_band[1], "-", anthro_band[2])
bio_tag <- paste0("bio_", bio_band[1], "-", bio_band[2])

output_csv <- file.path(site_dir, paste0(
  "ndsi_", gsub("[^A-Za-z0-9]+", "_", tolower(site_label)), "_",
  anthro_tag, "_", bio_tag, ".csv"
))

write.csv(all_results, output_csv, row.names = FALSE)
message("Saved to: ", output_csv)

summary_stats <- all_results %>%
  group_by(Site) %>%
  summarize(
    Segments  = n(),
    Mean_NDSI = mean(NDSI, na.rm = TRUE),
    SD_NDSI   = sd(NDSI, na.rm = TRUE),
    Mean_Anthro_Energy = mean(Anthro_Energy, na.rm = TRUE),
    SD_Anthro_Energy = sd(Anthro_Energy, na.rm = TRUE),
    Mean_Bio_Energy = mean(Bio_Energy, na.rm = TRUE),
    SD_Bio_Energy = sd(Bio_Energy, na.rm = TRUE)
  )
print(summary_stats)

plot_data <- all_results

p_ndsi <- ggplot(plot_data, aes(x = Time, y = NDSI)) +
  geom_line(size = 0.8, color = "#2C7FB8") +
  scale_x_datetime(
    date_labels = "%d-%b %H:%M",
    date_breaks = "2 hour",
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  labs(
    title = paste0("NDSI Over Time - ", site_label),
    x = "Date-Time",
    y = "NDSI"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.text.x = element_text(
      angle = 45,
      hjust = 1,
      vjust = 1,
      size = 6,
      margin = margin(t = 5)
    )
  )
print(p_ndsi)

p_anthro <- ggplot(plot_data, aes(x = Time, y = Anthro_Energy)) +
  geom_line(size = 0.8, color = "#D95F0E") +
  scale_x_datetime(
    date_labels = "%d-%b %H:%M",
    date_breaks = "2 hour",
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  labs(
    title = paste0("Anthropogenic Energy Over Time - ", site_label),
    x = "Date-Time",
    y = "Anthropogenic Energy (arbitrary units)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.text.x = element_text(
      angle = 45,
      hjust = 1,
      vjust = 1,
      size = 6,
      margin = margin(t = 5)
    )
  )
print(p_anthro)

p_bio <- ggplot(plot_data, aes(x = Time, y = Bio_Energy)) +
  geom_line(size = 0.8, color = "#1B9E77") +
  scale_x_datetime(
    date_labels = "%d-%b %H:%M",
    date_breaks = "2 hour",
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  labs(
    title = paste0("Biophonic Energy Over Time - ", site_label),
    x = "Date-Time",
    y = "Biophonic Energy (arbitrary units)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.text.x = element_text(
      angle = 45,
      hjust = 1,
      vjust = 1,
      size = 6,
      margin = margin(t = 5)
    )
  )
print(p_bio)

ndsi_quadrant_table <- plot_data %>%
  mutate(
    NDSI_Quadrant = case_when(
      NDSI >= 0.5  & NDSI <= 1   ~ "[0.5, 1]",
      NDSI >  0    & NDSI <  0.5 ~ "(0, 0.5)",
      NDSI >= -0.5 & NDSI <= 0   ~ "[-0.5, 0]",
      NDSI >= -1   & NDSI < -0.5 ~ "[-1, -0.5)",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(NDSI_Quadrant)) %>%
  group_by(Site, NDSI_Quadrant) %>%
  summarise(
    Segments = n(),
    .groups = "drop_last"
  ) %>%
  mutate(
    Total_Segments = sum(Segments),
    Percent_Time = 100 * Segments / Total_Segments
  ) %>%
  ungroup()

print(ndsi_quadrant_table)

beepr::beep(3)
