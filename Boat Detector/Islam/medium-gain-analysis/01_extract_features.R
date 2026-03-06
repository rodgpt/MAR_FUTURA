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

site_folders <- list(
  "Las Cruces 26"  = "Las Cruces 26",
  "Matanzas 32"    = "Matanzas 32",
  "San Antonio 38" = "San Antonio 38",
  "Ventanas 36"    = "Ventanas 36",
  "Zapallar 34"    = "Zapallar 34"
)

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

N_CORES <- max(1, detectCores() - 1)
cat(sprintf("Using %d cores\n\n", N_CORES))

all_results <- data.frame()

for (site_name in names(site_folders)) {
  folder   <- site_folders[[site_name]]
  wav_list <- list.files(folder, pattern="\\.wav$",
                          ignore.case=TRUE, full.names=FALSE)
  cat(sprintf("Processing: %-20s  %d files...\n", site_name, length(wav_list)))
  t0 <- proc.time()

  if (.Platform$OS.type == "windows") {
    cl <- makeCluster(N_CORES)
    clusterEvalQ(cl, { library(tuneR); library(signal) })
    clusterExport(cl, c("extract_features","folder"), envir=environment())
    results_list <- parLapply(cl, wav_list, function(f) {
      d <- extract_features(file.path(folder, f))
      if (!is.null(d)) cbind(file=f, d) else NULL
    })
    stopCluster(cl)
  } else {
    results_list <- mclapply(wav_list, function(f) {
      d <- extract_features(file.path(folder, f))
      if (!is.null(d)) cbind(file=f, d) else NULL
    }, mc.cores=N_CORES)
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
