suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
  library(tidyr)
  library(lubridate)
  library(ggplot2)
})

input_dir <- "/Users/Rodrigo/Library/CloudStorage/GoogleDrive-royanedel@marfutura.org/Unidades compartidas/MAR FUTURA/Hydrophones/NDSIResults/ndsi_casestudiesD2_20251115-20251116_anthro_1000-2000_bio_2000-3000.csv" # nolint: line_length_linter.

utils::globalVariables(c(
  "Site",
  "Time",
  "NDSI",
  "Anthro_Energy",
  "Bio_Energy",
  "NDSI_Quadrant",
  "Segments",
  "Total_Segments"
))

parse_args <- function(args) {
  out <- list(
    input_dir = NULL,
    input_file = NULL,
    output_dir = NULL,
    tz = "UTC"
  )

  if (length(args) == 0) return(out)

  key <- NULL
  for (a in args) {
    if (str_starts(a, "--")) {
      key <- sub("^--", "", a)
      next
    }
    if (!is.null(key)) {
      out[[key]] <- a
      key <- NULL
      next
    }
  }
  out
}

extract_run_meta <- function(path) {
  base <- tools::file_path_sans_ext(basename(path))
  m <- str_match(base, "^ndsi_casestudies_([0-9]{8}-[0-9]{8})_anthro_([0-9]+-[0-9]+)_bio_([0-9]+-[0-9]+)$")
  if (any(is.na(m))) {
    return(list(
      run_tag = base,
      date_tag = NA_character_,
      anthro_tag = NA_character_,
      bio_tag = NA_character_
    ))
  }
  list(
    run_tag = base,
    date_tag = m[2],
    anthro_tag = paste0("anthro_", m[3]),
    bio_tag = paste0("bio_", m[4])
  )
}

read_results_csv <- function(path, tz = "UTC") {
  df <- suppressMessages(read_csv(path, show_col_types = FALSE))

  required <- c("Site", "Time", "NDSI", "Anthro_Energy", "Bio_Energy")
  missing <- setdiff(required, names(df))
  if (length(missing) > 0) {
    stop("Missing columns in ", path, ": ", paste(missing, collapse = ", "))
  }

  df <- df %>%
    mutate(
      Site = as.character(Site),
      Time = as.POSIXct(Time, tz = tz)
    )

  if (any(is.na(df$Time))) {
    df <- df %>% mutate(Time = ymd_hms(Time, tz = tz, quiet = TRUE))
  }

  if (any(is.na(df$Time))) {
    stop("Could not parse Time column in ", path, ". Ensure it is an ISO datetime or something parseable by as.POSIXct/ymd_hms.")
  }

  df
}

make_plots <- function(plot_data, title_suffix = NULL) {
  if (!is.null(title_suffix) && nzchar(title_suffix)) {
    title_ndsi <- paste0("NDSI Over Time for Case Study Sites (", title_suffix, ")")
    title_anthro <- paste0("Anthropogenic Energy Over Time for Case Study Sites (", title_suffix, ")")
    title_bio <- paste0("Biophonic Energy Over Time for Case Study Sites (", title_suffix, ")")
  } else {
    title_ndsi <- "NDSI Over Time for Case Study Sites"
    title_anthro <- "Anthropogenic Energy Over Time for Case Study Sites"
    title_bio <- "Biophonic Energy Over Time for Case Study Sites"
  }

  p_ndsi <- ggplot(plot_data, aes(x = Time, y = NDSI, color = Site)) +
    geom_line(linewidth = 0.8) +
    facet_wrap(~Site, ncol = 1, scales = "free_x") +
    scale_x_datetime(
      date_labels = "%d-%b %H:%M",
      date_breaks = "2 hour",
      expand = expansion(mult = c(0.01, 0.01))
    ) +
    labs(
      title = title_ndsi,
      x = "Date-Time",
      y = "NDSI"
    ) +
    theme_minimal() +
    theme(
      legend.position = "none",
      plot.title = element_text(face = "bold", size = 14),
      axis.text.x = element_text(
        angle = 45,
        hjust = 1,
        vjust = 1,
        size = 6,
        margin = margin(t = 5)
      ),
      strip.text = element_text(face = "bold")
    )

  p_anthro <- ggplot(plot_data, aes(x = Time, y = Anthro_Energy, color = Site)) +
    geom_line(linewidth = 0.8) +
    facet_wrap(~Site, ncol = 1, scales = "free_x") +
    scale_x_datetime(
      date_labels = "%d-%b %H:%M",
      date_breaks = "2 hour",
      expand = expansion(mult = c(0.01, 0.01))
    ) +
    labs(
      title = title_anthro,
      x = "Date-Time",
      y = "Anthropogenic Energy (arbitrary units)"
    ) +
    theme_minimal() +
    theme(
      legend.position = "none",
      plot.title = element_text(face = "bold", size = 14),
      axis.text.x = element_text(
        angle = 45,
        hjust = 1,
        vjust = 1,
        size = 6,
        margin = margin(t = 5)
      ),
      strip.text = element_text(face = "bold")
    )

  p_bio <- ggplot(plot_data, aes(x = Time, y = Bio_Energy, color = Site)) +
    geom_line(linewidth = 0.8) +
    facet_wrap(~Site, ncol = 1, scales = "free_x") +
    scale_x_datetime(
      date_labels = "%d-%b %H:%M",
      date_breaks = "2 hour",
      expand = expansion(mult = c(0.01, 0.01))
    ) +
    labs(
      title = title_bio,
      x = "Date-Time",
      y = "Biophonic Energy (arbitrary units)"
    ) +
    theme_minimal() +
    theme(
      legend.position = "none",
      plot.title = element_text(face = "bold", size = 14),
      axis.text.x = element_text(
        angle = 45,
        hjust = 1,
        vjust = 1,
        size = 6,
        margin = margin(t = 5)
      ),
      strip.text = element_text(face = "bold")
    )

  list(p_ndsi = p_ndsi, p_anthro = p_anthro, p_bio = p_bio)
}

make_tables <- function(plot_data) {
  summary_stats <- plot_data %>%
    group_by(Site) %>%
    summarize(
      Segments = n(),
      Mean_NDSI = mean(NDSI, na.rm = TRUE),
      SD_NDSI = sd(NDSI, na.rm = TRUE),
      Mean_Anthro_Energy = mean(Anthro_Energy, na.rm = TRUE),
      SD_Anthro_Energy = sd(Anthro_Energy, na.rm = TRUE),
      Mean_Bio_Energy = mean(Bio_Energy, na.rm = TRUE),
      SD_Bio_Energy = sd(Bio_Energy, na.rm = TRUE),
      .groups = "drop"
    )

  ndsi_quadrant_table <- plot_data %>%
    mutate(
      NDSI_Quadrant = case_when(
        NDSI >= 0.5 & NDSI <= 1 ~ "[0.5, 1]",
        NDSI > 0 & NDSI < 0.5 ~ "(0, 0.5)",
        NDSI >= -0.5 & NDSI <= 0 ~ "[-0.5, 0]",
        NDSI >= -1 & NDSI < -0.5 ~ "[-1, -0.5)",
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

  list(summary_stats = summary_stats, ndsi_quadrant_table = ndsi_quadrant_table)
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))

  if ((is.null(args$input_dir) || !nzchar(args$input_dir)) && nzchar(input_dir)) {
    args$input_dir <- input_dir
  }

  if (!is.null(args$input_dir) && nzchar(args$input_dir) && file.exists(args$input_dir) && !dir.exists(args$input_dir)) {
    args$input_file <- args$input_dir
    args$input_dir <- NULL
  }

  input_paths <- character()
  if (!is.null(args$input_file) && nzchar(args$input_file)) {
    input_paths <- c(input_paths, args$input_file)
  }
  if (!is.null(args$input_dir) && nzchar(args$input_dir)) {
    input_paths <- c(input_paths, list.files(args$input_dir, pattern = "^ndsi_casestudies_.*\\.csv$", full.names = TRUE))
  }

  input_paths <- unique(input_paths)
  input_paths <- input_paths[file.exists(input_paths)]

  if (length(input_paths) == 0) {
    stop("No input files found. Provide --input_file <path> or --input_dir <dir> containing ndsi_casestudies_*.csv")
  }

  output_dir <- args$output_dir
  if (is.null(output_dir) || !nzchar(output_dir)) {
    if (!is.null(args$input_dir) && nzchar(args$input_dir)) {
      output_dir <- file.path(args$input_dir, "Graphs")
    } else {
      output_dir <- file.path(dirname(input_paths[[1]]), "Graphs")
    }
  }
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  for (path in input_paths) {
    meta <- extract_run_meta(path)
    message("Plotting: ", basename(path))

    plot_data <- read_results_csv(path, tz = args$tz)

    if ("Site" %in% names(plot_data)) {
      plot_data$Site <- factor(plot_data$Site, levels = unique(plot_data$Site))
    }

    plots <- make_plots(plot_data, title_suffix = meta$date_tag)
    tables <- make_tables(plot_data)

    base <- tools::file_path_sans_ext(basename(path))

    ggsave(filename = file.path(output_dir, paste0(base, "_ndsi.png")), plot = plots$p_ndsi, width = 10, height = 12, dpi = 200)
    ggsave(filename = file.path(output_dir, paste0(base, "_anthro_energy.png")), plot = plots$p_anthro, width = 10, height = 12, dpi = 200)
    ggsave(filename = file.path(output_dir, paste0(base, "_bio_energy.png")), plot = plots$p_bio, width = 10, height = 12, dpi = 200)

    write_csv(tables$summary_stats, file.path(output_dir, paste0(base, "_summary_stats.csv")))
    write_csv(tables$ndsi_quadrant_table, file.path(output_dir, paste0(base, "_ndsi_quadrant_table.csv")))
  }

  message("Done. Output in: ", normalizePath(output_dir))
}

main()















