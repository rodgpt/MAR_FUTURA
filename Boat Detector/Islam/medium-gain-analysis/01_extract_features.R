# ============================================================
# MARFUTURA BOAT DETECTION — STEP 1: FEATURE EXTRACTION
# Extracts spectral features from all WAV files across 5 sites
#
# Input:  WAV folders per site (60s, 16kHz, 16-bit mono)
# Output: boat_features_all.csv
# ============================================================

library(tuneR)
library(signal)
library(parallel)

if (exists("input_data_root") && exists("selected_site")) {
  site_folders <- setNames(list(input_data_root), selected_site)
} else {
  site_folders <- list(
    "Las Cruces 26"  = "Las Cruces 26",
    "Matanzas 32"    = "Matanzas 32",
    "San Antonio 38" = "San Antonio 38",
    "Ventanas 36"    = "Ventanas 36",
    "Zapallar 34"    = "Zapallar 34"
  )
}

extract_features <- function(wav_path) {
  tryCatch({
    sr  <- 16000
    hdr <- readWave(wav_path, header=TRUE)
    n   <- min(10*sr, hdr$samples)
    w   <- readWave(wav_path, from=1, to=n, units="samples")
    amp <- as.numeric(w@left)
    if (all(amp == 0)) return(NULL)

    wl  <- 1024; ol <- 512
    han <- hanning(wl)
    starts <- seq(1, length(amp)-wl, by=wl-ol)
    psd <- numeric(wl/2)
    for (i in seq_along(starts)) {
      ch  <- amp[starts[i]:(starts[i]+wl-1)] * han
      psd <- psd + Mod(fft(ch)[1:(wl/2)])^2
    }
    psd   <- psd / length(starts)
    freqs <- seq(0, sr/2, length.out=wl/2)

    bi <- which(freqs >= 100 & freqs <= 600 &
                !(freqs >= 520 & freqs <= 595))
    if (length(bi) < 10) return(NULL)

    psd_band <- psd[bi]
    f_band   <- freqs[bi]

    rms           <- sqrt(mean(amp^2))
    noise_floor   <- median(psd_band)
    boat_tonality <- 10*log10((mean(psd_band)+1e-10)/(noise_floor+1e-10))
    peak_freq     <- f_band[which.max(psd_band)]

    sorted_psd  <- sort(psd_band, decreasing=TRUE)
    peak_thresh <- sorted_psd[min(5, length(sorted_psd))]
    peak_idx    <- which(psd_band >= peak_thresh)
    peak_freqs_found <- f_band[peak_idx]
    n_harmonic_peaks <- 0; last_peak <- -999
    for (pf in sort(peak_freqs_found)) {
      if (pf - last_peak >= 20) {
        n_harmonic_peaks <- n_harmonic_peaks + 1
        last_peak <- pf
      }
    }

    p_norm           <- psd_band / (sum(psd_band)+1e-10)
    spectral_entropy <- -sum(p_norm * log(p_norm+1e-10))
    band_energy_ratio <- sum(psd_band) / (sum(psd)+1e-10)

    data.frame(
      rms               = round(rms, 4),
      boat_tonality     = round(boat_tonality, 3),
      peak_freq         = round(peak_freq, 1),
      n_harmonic_peaks  = n_harmonic_peaks,
      spectral_entropy  = round(spectral_entropy, 3),
      band_energy_ratio = round(band_energy_ratio, 5),
      stringsAsFactors  = FALSE
    )
  }, error=function(e) NULL)
}

split_into_chunks <- function(x, chunk_size) {
  if (length(x) == 0) return(list())
  split(x, ceiling(seq_along(x) / chunk_size))
}

N_CORES <- max(1, detectCores() - 1)
progress_mode <- exists("input_data_root") && exists("selected_site")
if (progress_mode) {
  cat("Using sequential mode for visible progress updates\n\n")
} else {
  cat(sprintf("Using %d cores\n\n", N_CORES))
}

all_results <- data.frame()

for (site_name in names(site_folders)) {
  folder   <- site_folders[[site_name]]
  wav_list <- list.files(folder, pattern="\\.wav$",
                          ignore.case=TRUE, full.names=FALSE)
  cat(sprintf("Processing: %-20s  %d files...\n", site_name, length(wav_list)))
  t0 <- proc.time()

  if (length(wav_list) == 0) {
    cat("  No WAV files found\n")
    next
  }

  chunk_size <- max(1, min(250, ceiling(length(wav_list) / 100)))
  wav_chunks <- split_into_chunks(wav_list, chunk_size)
  processed_count <- 0
  last_reported_count <- 0
  results_list <- list()

  if (progress_mode) {
    report_every <- 10
    print(sprintf("Progress: 0 / %d files", length(wav_list)))
    flush.console()
    for (f in wav_list) {
      d <- extract_features(file.path(folder, f))
      if (!is.null(d)) {
        results_list[[length(results_list) + 1]] <- cbind(file=f, d)
      }
      processed_count <- processed_count + 1
      if (processed_count >= length(wav_list) ||
          processed_count - last_reported_count >= report_every) {
        print(sprintf("Progress: %d / %d files", processed_count, length(wav_list)))
        flush.console()
        last_reported_count <- processed_count
      }
    }
  } else if (.Platform$OS.type == "windows") {
    cl <- makeCluster(N_CORES)
    clusterEvalQ(cl, { library(tuneR); library(signal) })
    clusterExport(cl, c("extract_features","folder"), envir=environment())
    for (chunk in wav_chunks) {
      chunk_results <- parLapply(cl, chunk, function(f) {
        d <- extract_features(file.path(folder, f))
        if (!is.null(d)) cbind(file=f, d) else NULL
      })
      results_list <- c(results_list, chunk_results)
      processed_count <- processed_count + length(chunk)
      if (processed_count >= length(wav_list) ||
          processed_count - last_reported_count >= chunk_size) {
        cat(sprintf("  Progress: %d / %d files\n", processed_count, length(wav_list)))
        flush.console()
        last_reported_count <- processed_count
      }
    }
    stopCluster(cl)
  } else {
    for (chunk in wav_chunks) {
      chunk_results <- mclapply(chunk, function(f) {
        d <- extract_features(file.path(folder, f))
        if (!is.null(d)) cbind(file=f, d) else NULL
      }, mc.cores=N_CORES)
      results_list <- c(results_list, chunk_results)
      processed_count <- processed_count + length(chunk)
      if (processed_count >= length(wav_list) ||
          processed_count - last_reported_count >= chunk_size) {
        cat(sprintf("  Progress: %d / %d files\n", processed_count, length(wav_list)))
        flush.console()
        last_reported_count <- processed_count
      }
    }
  }

  site_df <- do.call(rbind, Filter(Negate(is.null), results_list))
  if (!is.null(site_df) && nrow(site_df) > 0) {
    site_df$site <- site_name
    all_results  <- rbind(all_results, site_df)
  }

  dt <- (proc.time() - t0)["elapsed"]
  cat(sprintf("  Done in %.0fs (%.0f files/sec)\n", dt, length(wav_list)/dt))
}

all_results$datetime_utc   <- as.POSIXct(
  sub("\\.[wW][aA][vV]$","", all_results$file),
  format="%Y%m%d_%H%M%S", tz="UTC")
all_results$datetime_chile <- all_results$datetime_utc - 3*3600
all_results$hour_chile     <- as.integer(format(all_results$datetime_chile, "%H"))
all_results$date_chile     <- as.Date(all_results$datetime_chile)
all_results$is_night       <- all_results$hour_chile >= 22 |
                               all_results$hour_chile <= 6

write.csv(all_results, "boat_features_all.csv", row.names=FALSE)
cat(sprintf("\nSaved: boat_features_all.csv (%d files)\n", nrow(all_results)))
