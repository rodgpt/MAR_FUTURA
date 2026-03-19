# ============================================================
# MARFUTURA BOAT DETECTION — STEP 3: EVENT REPORTING
#
# Reconstructs vessel events from consecutive detection files.
# Each event = unbroken run of flagged consecutive 1-min files.
# ============================================================

all_features <- read.csv("boat_features_all.csv", stringsAsFactors=FALSE)

all_features$datetime_utc   <- as.POSIXct(
  sub("\\.[wW][aA][vV]$","", all_features$file),
  format="%Y%m%d_%H%M%S", tz="UTC")
all_features$datetime_chile <- all_features$datetime_utc - 3*3600

report_events <- function(site_name, all_features) {
  s   <- all_features[all_features$site == site_name, ]
  s   <- s[order(s$datetime_chile), ]
  idx <- which(s$boat_detected)
  if (length(idx) == 0) return(data.frame())

  runs <- list(); run <- c(idx[1])
  if (length(idx) > 1) {
    for (i in 2:length(idx)) {
      if (idx[i]==idx[i-1]+1) { run <- c(run, idx[i])
      } else { runs[[length(runs)+1]] <- run; run <- c(idx[i]) }
    }
  }
  runs[[length(runs)+1]] <- run

  out <- data.frame()
  for (r in runs) {
    t0   <- s$datetime_chile[r[1]]
    t1   <- s$datetime_chile[r[length(r)]]
    dur  <- as.numeric(difftime(t1, t0, units="mins")) + 1
    best <- which.max(s$boat_tonality[r])
    out  <- rbind(out, data.frame(
      site         = site_name,
      date_chile   = as.Date(t0),
      time_chile   = format(t0, "%H:%M"),
      time_utc     = format(s$datetime_utc[r[1]], "%H:%M"),
      duration_min = round(dur),
      n_files      = length(r),
      max_tonality = round(max(s$boat_tonality[r], na.rm=TRUE)),
      peak_freq_hz = round(s$peak_freq[r[best]]),
      stringsAsFactors=FALSE))
  }
  out
}

sites <- sort(unique(all_features$site))

all_events <- data.frame()
for (site_name in sites) {
  ev <- report_events(site_name, all_features)
  if (nrow(ev) > 0) all_events <- rbind(all_events, ev)
  cat(sprintf("%s: %d events\n", site_name, nrow(ev)))
  if (nrow(ev) > 0) print(ev, row.names=FALSE)
  cat("\n")
}

cat(sprintf("TOTAL: %d nighttime vessel events\n\n", nrow(all_events)))

dir.create("outputs", showWarnings=FALSE)
write.csv(all_events, "outputs/vessel_events_FINAL.csv", row.names=FALSE)
cat("Saved: outputs/vessel_events_FINAL.csv\n")
