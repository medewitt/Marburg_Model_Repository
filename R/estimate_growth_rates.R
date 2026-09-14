# Intrinsic growth rate of each Marburg outbreak from the ascending phase of the
# zero-filled daily incidence series, with a NEGATIVE BINOMIAL observation model.
#
# Daily case counts are overdispersed relative to the Poisson mean = variance
# assumption, so Poisson standard errors are optimistic. We fit a negative
# binomial (NB2) log-linear model as the primary estimate,
#
#   MASS::glm.nb(cases ~ t)      # Var = mu + alpha * mu^2,  alpha = 1/theta
#
# and keep the Poisson fit as a sensitivity analysis. r is the slope on t
# (t = onset/report day - outbreak start); 95% Wald CI and doubling time
# ln(2)/r are reported. A likelihood-ratio test of Poisson vs NB (boundary
# 0.5 chi^2_1) gives overdisp_p; small values indicate genuine overdispersion.
#
# Where the counts are too sparse to identify the NB dispersion (glm.nb fails to
# converge or returns an unstable SE), the NB columns fall back to the Poisson
# fit and the note records it. The linear pseudo-R0 columns of the previous
# version are dropped; R0 is now in outbreak_R0_montecarlo.csv (gamma
# generation interval, Monte Carlo).
#
# Method: (1) build a complete daily series and zero-fill; (2) restrict to day 0
# through the LAST day at peak daily incidence; (3) outbreaks with peak <= 1 or
# a growth phase < 3 days are NA. Seed 1834 kept for repo convention (fitting is
# deterministic).

set.seed(1834)
Sys.setlocale("LC_TIME", "C")
suppressPackageStartupMessages(library(MASS))  # glm.nb

z <- qnorm(0.975)

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
       kind = "line", drop_index = TRUE),
  list(label = "Rwanda 2024",    file = "MarburgRwanda2024Line.csv",  kind = "line")
)

NOTES <- c(
  "Tanzania 2023" = "all cases; r inflated by 9-day gap between index case (onset 27 Feb) and cluster",
  "Tanzania 2023 (excl. index)" = "index case (onset 27 Feb) dropped; post-introduction human-to-human phase",
  "Rwanda 2024" = "66 lab-confirmed cases by onset, digitized from NEJM Fig 1 (Nsanzimana 2025); 2 probable Aug cases excluded"
)

daily_series <- function(file, kind, drop_index = FALSE) {
  df <- read.csv(file.path("..", "data", file), check.names = FALSE,
                 stringsAsFactors = FALSE)
  if (kind == "agg") {
    d <- as.Date(df[["Reported date"]], format = "%d/%m/%Y")
    c <- suppressWarnings(as.numeric(df[["new cases"]]))
    ok <- !is.na(d) & !is.na(c)
    agg <- aggregate(c[ok], by = list(date = d[ok]), FUN = sum)
    names(agg)[2] <- "cases"
    return(agg[order(agg$date), ])
  }
  d <- if (kind == "drc") as.Date(df[["DT_ONSET"]], format = "%d-%b-%y")
       else as.Date(df[["ONSET_DATE"]], format = "%d/%m/%Y")
  d <- d[!is.na(d)]
  if (drop_index && length(d)) d <- d[d != min(d)]
  if (length(d) == 0) return(data.frame(date = as.Date(character()), cases = integer()))
  full <- seq(min(d), max(d), by = "day")
  data.frame(date = full, cases = as.integer(table(factor(d, levels = as.character(full)))))
}

blank_row <- function(label, total, ndf) data.frame(
  outbreak = label, total_cases = total, n_days_full = ndf,
  n_days_growth_phase = NA_integer_, cases_growth_phase = NA_integer_,
  r_nb = NA_real_, se_nb = NA_real_, ci_low_nb = NA_real_, ci_high_nb = NA_real_,
  doubling_nb = NA_real_, nb_alpha = NA_real_, overdisp_p = NA_real_,
  r_poisson = NA_real_, se_poisson = NA_real_, ci_low_poisson = NA_real_,
  ci_high_poisson = NA_real_, note = "", stringsAsFactors = FALSE)

res <- data.frame()
for (ob in outbreaks) {
  s <- daily_series(ob$file, ob$kind, drop_index = isTRUE(ob$drop_index))
  total <- sum(s$cases)
  note_kind <- paste(Filter(nzchar, c(
    if (ob$kind == "agg") "aggregate surveillance (irregular reporting; new cases per report)" else "",
    if (ob$label %in% names(NOTES)) NOTES[[ob$label]] else "")), collapse = "; ")
  row <- blank_row(ob$label, total, nrow(s))

  if (nrow(s) == 0) { row$note <- "no parseable dates"; res <- rbind(res, row); next }
  peak_val <- max(s$cases); peak_pos <- max(which(s$cases == peak_val))
  win <- s[seq_len(peak_pos), ]
  row$n_days_growth_phase <- nrow(win); row$cases_growth_phase <- sum(win$cases)
  if (peak_val < 2 || nrow(win) < 3) {
    reason <- if (peak_val < 2) "point-source / no growth phase: peak daily incidence <= 1"
              else "growth phase < 3 days"
    row$note <- paste(Filter(nzchar, c(note_kind, reason)), collapse = "; ")
    res <- rbind(res, row); next
  }

  d <- data.frame(cases = win$cases, t = as.numeric(win$date - min(win$date)))
  pois <- glm(cases ~ t, family = poisson(), data = d)
  rp <- coef(pois)[["t"]]; sep <- sqrt(vcov(pois)["t", "t"])
  row$r_poisson <- round(rp, 4) + 0; row$se_poisson <- round(sep, 4)
  row$ci_low_poisson <- round(rp - z * sep, 4) + 0
  row$ci_high_poisson <- round(rp + z * sep, 4) + 0

  nb <- tryCatch(suppressWarnings(glm.nb(cases ~ t, data = d)), error = function(e) NULL)
  ok <- !is.null(nb) && is.finite(sqrt(vcov(nb)["t", "t"])) && sqrt(vcov(nb)["t", "t"]) <= 10
  if (ok) {
    rnb <- coef(nb)[["t"]]; senb <- sqrt(vcov(nb)["t", "t"])
    lr <- as.numeric(2 * (logLik(nb) - logLik(pois)))
    row$r_nb <- round(rnb, 4) + 0; row$se_nb <- round(senb, 4)
    row$ci_low_nb <- round(rnb - z * senb, 4) + 0
    row$ci_high_nb <- round(rnb + z * senb, 4) + 0
    row$doubling_nb <- if (rnb > 0) round(log(2) / rnb, 1) else NA_real_
    row$nb_alpha <- round(1 / nb$theta, 4)        # alpha = 1/theta (NB2)
    row$overdisp_p <- round(0.5 * pchisq(max(lr, 0), 1, lower.tail = FALSE), 4)
    row$note <- note_kind
  } else {
    row$r_nb <- round(rp, 4) + 0; row$se_nb <- round(sep, 4)
    row$ci_low_nb <- round(rp - z * sep, 4) + 0
    row$ci_high_nb <- round(rp + z * sep, 4) + 0
    row$doubling_nb <- if (rp > 0) round(log(2) / rp, 1) else NA_real_
    row$note <- paste(Filter(nzchar, c(note_kind, "NB alpha not identified; NB columns = Poisson fit")),
                      collapse = "; ")
  }
  res <- rbind(res, row)
}

res <- res[, c(setdiff(names(res), "note"), "note")]
write.csv(res, "../outputs/outbreak_growth_rates.csv", row.names = FALSE, na = "")
print(res)
