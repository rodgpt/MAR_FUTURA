# ============================================================
# MARFUTURA BOAT DETECTION — STEP 2: DETECTION RULE
#
# Applies rule-based detector to extracted features.
# Nighttime only (22:00-06:00 Chile local = UTC-3)
# ============================================================

library(tuneR)
library(signal)

all_features <- read.csv("boat_features_all.csv", stringsAsFactors=FALSE)

if (!"datetime_chile" %in% names(all_features)) {
  all_features$datetime_utc   <- as.POSIXct(
    sub("\\.[wW][aA][vV]$","", all_features$file),
    format="%Y%m%d_%H%M%S", tz="UTC")
  all_features$datetime_chile <- all_features$datetime_utc - 3*3600
  all_features$hour_chile     <- as.integer(format(all_features$datetime_chile, "%H"))
  all_features$date_chile     <- as.Date(all_features$datetime_chile)
  all_features$is_night       <- all_features$hour_chile >= 22 |
                                  all_features$hour_chile <= 6
}

# ── Site parameters ──────────────────────────────────────────
default_site_param <- list(tonality=30, min_cluster=1, low_freq_peaks=2)

site_params <- list(
  "Las Cruces 26"  = list(tonality=30, min_cluster=1, low_freq_peaks=2),
  "Matanzas 32"    = list(tonality=30, min_cluster=1, low_freq_peaks=2),
  "San Antonio 38" = list(tonality=25, min_cluster=2, low_freq_peaks=2),
  "Ventanas 36"    = list(tonality=25, min_cluster=2, low_freq_peaks=2),
  "Zapallar 34"    = list(tonality=30, min_cluster=1, low_freq_peaks=2)
)

sites <- sort(unique(all_features$site))
for (site_name in sites) {
  if (is.null(site_params[[site_name]])) {
    site_params[[site_name]] <- default_site_param
  }
}

FREQ_MIN    <- 100;  FREQ_MAX    <- 600
INTERF_LOW  <- 520;  INTERF_HIGH <- 595

# ── Cluster filter ───────────────────────────────────────────
apply_cluster_filter <- function(site_name, features, flag_col, min_size) {
  s    <- features[features$site == site_name, ]
  s    <- s[order(s$datetime_chile), ]
  idx  <- which(s[[flag_col]])
  keep <- rep(FALSE, nrow(s))
  if (length(idx) == 0) return(keep)
  if (length(idx) == 1) {
    if (min_size <= 1) keep[idx] <- TRUE
    return(keep)
  }
  runs <- list(); run <- c(idx[1])
  for (i in 2:length(idx)) {
    if (idx[i] == idx[i-1]+1) { run <- c(run, idx[i])
    } else { runs[[length(runs)+1]] <- run; run <- c(idx[i]) }
  }
  runs[[length(runs)+1]] <- run
  for (r in runs) if (length(r) >= min_size) keep[r] <- TRUE
  keep
}

# ── Apply rule ───────────────────────────────────────────────
all_features$rule_base <- FALSE

for (site_name in sites) {
  p   <- site_params[[site_name]]
  idx <- which(all_features$site == site_name)
  s   <- all_features[idx, ]

  base <- with(s,
    !is.na(boat_tonality)                                    &
    boat_tonality    >  p$tonality                           &
    peak_freq        >= FREQ_MIN                             &
    peak_freq        <= FREQ_MAX                             &
    n_harmonic_peaks >= 1                                    &
    !(peak_freq >= INTERF_LOW & peak_freq <= INTERF_HIGH)    &
    !(peak_freq < 200 & n_harmonic_peaks < p$low_freq_peaks)
  )

  # Matanzas: exclude biological sound band 100-135 Hz
  if (site_name == "Matanzas 32") {
    base <- base & !(s$peak_freq >= 100 & s$peak_freq <= 135)
  }

  all_features$rule_base[idx] <- base
}

all_features$boat_detected <- FALSE
for (site_name in sites) {
  p    <- site_params[[site_name]]
  idx  <- which(all_features$site == site_name)
  keep <- apply_cluster_filter(site_name, all_features,
                                "rule_base", p$min_cluster)
  all_features$boat_detected[idx] <- keep
}

# ── Summary ──────────────────────────────────────────────────
cat("============================================================\n")
cat("RESULTS — All hours\n")
cat("============================================================\n\n")
cat(sprintf("%-20s  %7s  %7s  %6s\n","Site","Hours","Events","Files"))
cat(strrep("-", 46), "\n")

total_events <- 0
for (site_name in sites) {
  s   <- all_features[all_features$site==site_name, ]
  hrs <- round(nrow(s)*60/3600, 1)
  det <- sum(s$boat_detected, na.rm=TRUE)
  s2  <- s[order(s$datetime_chile), ]
  di  <- which(s2$boat_detected)
  n_ev <- if (length(di)==0) 0 else {
    ev <- 1
    if (length(di)>1)
      for (i in 2:length(di))
        if (di[i]!=di[i-1]+1) ev <- ev+1
    ev
  }
  total_events <- total_events + n_ev
  cat(sprintf("%-20s  %7.1fh  %7d  %6d\n", site_name, hrs, n_ev, det))
}
cat(strrep("-", 46), "\n")
cat(sprintf("%-20s  %7.1fh  %7d  %6d\n",
    "TOTAL", nrow(all_features)*60/3600,
    total_events, sum(all_features$boat_detected, na.rm=TRUE)))

all_features$time_chile <- format(all_features$datetime_chile, "%H:%M")

dir.create("outputs", showWarnings=FALSE)
det_out <- all_features[all_features$boat_detected==TRUE,
  c("site","file","date_chile","time_chile","hour_chile",
    "boat_tonality","n_harmonic_peaks","peak_freq")]
write.csv(det_out, "outputs/boat_detections_FINAL.csv", row.names=FALSE)
cat("\nSaved: outputs/boat_detections_FINAL.csv\n")
