# Estimate the intrinsic growth rate of each Marburg outbreak with a
# Poisson log-linear model: glm(cases ~ t, family = poisson()), where
# t = report/onset date - start of outbreak (in days).
#
# Method (see the note column of the output for per-outbreak caveats):
#   1. Build a complete daily incidence series per outbreak and zero-fill
#      days with no reported cases.
#   2. Restrict to the ascending (growth) phase: day 0 through the LAST day
#      that attains the maximum daily incidence. This is the phase over which
#      the intrinsic growth rate r is defined (I(t) proportional to exp(r t)).
#   3. Fit glm(cases ~ t, family = poisson()); r is the slope on t.
#      Report r with a large-sample (Wald) 95% CI and doubling time ln(2)/r.
#   4. Outbreaks with peak daily incidence <= 1 or a growth phase < 3 days
#      are point-source / too small to identify r and are reported as NA.
#   5. Add a pseudo basic reproduction number R0 = 1 + r * SI, the
#      Wallinga & Lipsitch (2007) result for an exponentially distributed
#      generation interval, using the serial interval SI as a proxy. Pre-2014
#      Marburg serial-interval estimates of 9.3 and 11.2 days are used; the
#      95% CI is propagated from the growth-rate Wald CI.
#
# Data formats: Angola 2005 is aggregate surveillance (new cases per reported
# date); all other files are line lists keyed on symptom-onset date
# (ONSET_DATE, or DT_ONSET for DRC 1999). Poisson GLM fitting (IRLS) is
# deterministic, so no random seed is required; set.seed(1834) is kept for
# convention with the rest of the repository.

set.seed(1834)
Sys.setlocale("LC_TIME", "C")  # ensure English month abbreviations for DRC dates

z <- qnorm(0.975)
SI <- c("9.3" = 9.3, "11.2" = 11.2)  # pre-2014 mean serial intervals (days)

outbreaks <- list(
  list(label = "Angola 2005",      file = "MarburgAngola2005Data.csv",  kind = "agg"),
  list(label = "Belgrade 1967",    file = "MarburgBelgrade1967Line.csv",kind = "line"),
  list(label = "DRC 1999",         file = "MarburgDRC1999LineData.csv", kind = "drc"),
  list(label = "Frankfurt 1967",   file = "MarburgFrankfurt1967Line.csv",kind = "line"),
  list(label = "Guinea 2021",      file = "MarburgGuinea2021Line.csv",  kind = "line"),
  list(label = "Kenya 1980",       file = "MarburgKenya1980Line.csv",   kind = "line"),
  list(label = "Kenya 1987",       file = "MarburgKenya1987Line.csv",   kind = "line"),
  list(label = "Marburg 1967",     file = "MarburgMarburg1967Line.csv", kind = "line"),
  list(label = "South Africa 1975",file = "MarburgSA1975Line.csv",      kind = "line"),
  list(label = "Uganda (ND) 2008", file = "MarburgUG_ND2008Line.csv",   kind = "line"),
  list(label = "Uganda (US) 2008", file = "MarburgUG_US2008Line.csv",   kind = "line"),
  list(label = "Uganda 2007",      file = "MarburgUganda2007Line.csv",  kind = "line"),
  list(label = "Uganda 2012",      file = "MarburgUganda2012Line.csv",  kind = "line"),
  list(label = "Uganda 2014",      file = "MarburgUganda2014Line.csv",  kind = "line"),
  list(label = "Uganda 2017",      file = "MarburgUganda2017Line.csv",  kind = "line"),
  list(label = "Tanzania 2023",    file = "MarburgTanzania2023Line.csv",kind = "line"),
  list(label = "Tanzania 2023 (excl. index)", file = "MarburgTanzania2023Line.csv",
       kind = "line", drop_index = TRUE)
)

# explicit per-outbreak notes
NOTES <- c(
  "Tanzania 2023" = "all cases; r inflated by 9-day gap between index case (onset 27 Feb) and cluster",
  "Tanzania 2023 (excl. index)" = "index case (onset 27 Feb) dropped; post-introduction human-to-human phase"
)

# Return a data.frame(date, cases) of the complete zero-filled daily series.
daily_series <- function(file, kind, drop_index = FALSE) {
  df <- read.csv(file, check.names = FALSE, stringsAsFactors = FALSE)
  if (kind == "agg") {
    d <- as.Date(df[["Reported date"]], format = "%d/%m/%Y")
    c <- suppressWarnings(as.numeric(df[["new cases"]]))
    ok <- !is.na(d) & !is.na(c)
    agg <- aggregate(c[ok], by = list(date = d[ok]), FUN = sum)
    names(agg)[2] <- "cases"
    return(agg[order(agg$date), ])
  }
  if (kind == "drc") {
    d <- as.Date(df[["DT_ONSET"]], format = "%d-%b-%y")
  } else {
    d <- as.Date(df[["ONSET_DATE"]], format = "%d/%m/%Y")
  }
  d <- d[!is.na(d)]
  if (drop_index && length(d)) d <- d[d != min(d)]  # drop index case(s) at earliest onset
  if (length(d) == 0) return(data.frame(date = as.Date(character()), cases = integer()))
  full <- seq(min(d), max(d), by = "day")
  counts <- as.integer(table(factor(d, levels = as.character(full))))
  data.frame(date = full, cases = counts)
}

res <- data.frame()
for (ob in outbreaks) {
  drop_index <- isTRUE(ob$drop_index)
  s <- daily_series(ob$file, ob$kind, drop_index = drop_index)
  total <- sum(s$cases)
  note_parts <- c(if (ob$kind == "agg")
    "aggregate surveillance (irregular reporting; new cases per report)" else "",
    if (ob$label %in% names(NOTES)) NOTES[[ob$label]] else "")
  note_kind <- paste(note_parts[nzchar(note_parts)], collapse = "; ")

  row <- data.frame(outbreak = ob$label, total_cases = total,
                    n_days_full = nrow(s), n_days_growth_phase = NA_integer_,
                    cases_growth_phase = NA_integer_, growth_rate_per_day = NA_real_,
                    std_error = NA_real_, ci_low = NA_real_, ci_high = NA_real_,
                    doubling_time_days = NA_real_, note = "", stringsAsFactors = FALSE,
                    check.names = FALSE)
  for (k in names(SI))
    row[[paste0("pseudoR0_SI", k)]] <- row[[paste0("pseudoR0_SI", k, "_low")]] <-
      row[[paste0("pseudoR0_SI", k, "_high")]] <- NA_real_

  if (nrow(s) == 0) { row$note <- "no parseable dates"; res <- rbind(res, row); next }

  peak_val <- max(s$cases)
  peak_pos <- max(which(s$cases == peak_val))   # LAST day at maximum incidence
  win <- s[seq_len(peak_pos), ]
  row$n_days_growth_phase <- nrow(win)
  row$cases_growth_phase  <- sum(win$cases)

  if (peak_val < 2 || nrow(win) < 3) {
    reason <- if (peak_val < 2)
      "point-source / no growth phase: peak daily incidence <= 1" else "growth phase < 3 days"
    row$note <- paste(c(note_kind, reason)[nzchar(c(note_kind, reason))], collapse = "; ")
    res <- rbind(res, row); next
  }

  t <- as.numeric(win$date - min(win$date))
  m <- glm(cases ~ t, family = poisson(), data = data.frame(cases = win$cases, t = t))
  r  <- coef(m)[["t"]]
  se <- sqrt(vcov(m)["t", "t"])
  row$growth_rate_per_day <- round(r, 4) + 0    # + 0 normalizes -0 to 0
  row$std_error <- round(se, 4)
  row$ci_low  <- round(r - z * se, 4) + 0
  row$ci_high <- round(r + z * se, 4) + 0
  dt <- if (r > 0) log(2) / r else NA_real_
  row$doubling_time_days <- if (r > 0) round(dt, 1) else NA_real_
  for (k in names(SI)) {
    si <- SI[[k]]
    row[[paste0("pseudoR0_SI", k)]]      <- round(1 + r * si, 2)
    row[[paste0("pseudoR0_SI", k, "_low")]]  <- round(1 + (r - z * se) * si, 2)
    row[[paste0("pseudoR0_SI", k, "_high")]] <- round(1 + (r + z * se) * si, 2)
  }
  row$note <- note_kind
  res <- rbind(res, row)
}

# order columns with note last
r0cols <- unlist(lapply(names(SI), function(k)
  paste0("pseudoR0_SI", k, c("", "_low", "_high"))))
res <- res[, c(setdiff(names(res), c("note", r0cols)), r0cols, "note")]
write.csv(res, "outbreak_growth_rates.csv", row.names = FALSE, na = "")
print(res)
