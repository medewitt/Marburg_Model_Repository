# Monte Carlo pseudo R0 with uncertainty in the generation interval.
#
# For each outbreak with an estimable growth rate (from outbreak_growth_rates.csv)
# we propagate three sources of uncertainty to R0:
#   1. growth rate      r  ~ Normal(r_hat, se_r)                (GLM Wald)
#   2. mean gen. interval T_g ~ Normal(mu_era, sd_tg_mean), truncated > 0,
#      with era-specific means: pre-2014 outbreaks 9.3 d, 2020s outbreaks 11 d
#      (the reported shift in the serial-interval estimate);
#   3. GI dispersion via the coefficient of variation
#      CV ~ Normal(0.6, 0.1) truncated to [0.3, 1.0].
#
# R0 uses the gamma generation-interval form (Wallinga & Lipsitch 2007):
#   R0 = (1 + r * T_g / kappa)^kappa,  kappa = 1/CV^2 = (T_g / SD_GI)^2.
# CV = 1 recovers the exponential-GI result 1 + r*T_g; CV -> 0 recovers
# exp(r*T_g). R0 is undefined when the base is <= 0 (strong decline); such
# draws are dropped and their fraction reported as pct_undefined.
#
# All assumptions are the editable constants below. Seed 1834 for reproducibility.

set.seed(1834)
N <- 200000L
SD_TG_MEAN <- 1.0                 # uncertainty in the era mean generation interval (days)
TG_MEAN <- c(pre2014 = 9.3, "2020s" = 11.0)
CV_MEAN <- 0.6; CV_SD <- 0.1; CV_LO <- 0.3; CV_HI <- 1.0
Z <- qnorm(0.975)

rtnorm <- function(n, mean, sd, lo = -Inf, hi = Inf) {
  x <- rnorm(n, mean, sd)
  repeat {
    bad <- x < lo | x > hi
    if (!any(bad)) break
    x[bad] <- rnorm(sum(bad), mean, sd)
  }
  x
}

era_of <- function(label) {
  yr <- as.integer(sub(".*?([0-9]{4}).*", "\\1", label))
  if (is.na(yr)) NA_character_ else if (yr < 2014) "pre2014" else "2020s"
}

gr <- read.csv("outbreak_growth_rates.csv", check.names = FALSE, stringsAsFactors = FALSE)
gr <- gr[!is.na(gr$growth_rate_per_day), ]

out <- data.frame()
for (i in seq_len(nrow(gr))) {
  label <- gr$outbreak[i]; rh <- gr$growth_rate_per_day[i]; se <- gr$std_error[i]
  era <- era_of(label); mu <- TG_MEAN[[era]]
  r  <- rnorm(N, rh, se)
  tg <- rtnorm(N, mu, SD_TG_MEAN, lo = 0.1, hi = 50)
  cv <- rtnorm(N, CV_MEAN, CV_SD, lo = CV_LO, hi = CV_HI)
  kappa <- 1 / cv^2
  base <- 1 + r * tg / kappa
  R0 <- ifelse(base > 0, base^kappa, NA_real_)
  q <- quantile(R0, c(0.025, 0.5, 0.975), na.rm = TRUE)
  out <- rbind(out, data.frame(
    outbreak = label, era = era, r_per_day = rh, se_r = se,
    r_lo = round(rh - Z * se, 4), r_hi = round(rh + Z * se, 4),
    tg_mean_days = mu, sd_tg_mean = SD_TG_MEAN,
    tg_lo = round(mu - Z * SD_TG_MEAN, 2), tg_hi = round(mu + Z * SD_TG_MEAN, 2),
    cv_prior = sprintf("N(%.1f,%.1f) trunc[%.1f,%.1f]", CV_MEAN, CV_SD, CV_LO, CV_HI),
    R0_gamma_median = round(q[[2]], 2), R0_gamma_lo = round(q[[1]], 2),
    R0_gamma_hi = round(q[[3]], 2),
    pct_undefined = round(mean(is.na(R0)) * 100, 1),
    stringsAsFactors = FALSE))
}
write.csv(out, "outbreak_R0_montecarlo.csv", row.names = FALSE)
print(out[, c("outbreak","era","tg_mean_days","R0_gamma_median","R0_gamma_lo","R0_gamma_hi","pct_undefined")])
