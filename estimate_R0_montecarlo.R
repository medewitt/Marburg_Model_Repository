# Monte Carlo pseudo R0 with an outbreak-informed generation interval.
#
# For each outbreak with an estimable growth rate we propagate:
#   1. growth rate   r  ~ Normal(r_hat, se_r)  (GLM Wald);
#   2. mean generation interval T_g ~ Normal(mu, se_mu), truncated > 0, where
#      mu is the mean serial interval of that outbreak's reconstructed
#      transmission pairs when it has >= 3 (se_mu = pair SD / sqrt(n)),
#      otherwise the pooled Marburg estimate mu = 9.2 d (Qian et al. 2023,
#      medRxiv 10.1101/2022.06.17.22276538; se ~ 0.7).
#   3. GI dispersion: the coefficient of variation is fixed at the pooled
#      literature value CV = 4.4/9.2 = 0.478 for ALL outbreaks. Per-outbreak
#      pair samples are star-shaped (most share one infector) so their SD is
#      not a reliable dispersion; only the mean is taken from the pairs.
#
# R0 = (1 + r*T_g/kappa)^kappa, kappa = 1/CV^2 (gamma generation interval).
# Undefined when the base <= 0 (strong decline); such draws are dropped.
# Seed 1834; N = 2e5.

set.seed(1834)
N <- 200000L
POOL_MEAN <- 9.2; POOL_SE <- 0.7; POOL_CV <- round(4.4 / 9.2, 3)  # 0.478
MIN_PAIRS <- 3L
Z <- qnorm(0.975)
KAPPA <- 1 / POOL_CV^2

rtnorm <- function(n, mean, sd, lo = -Inf, hi = Inf) {
  x <- rnorm(n, mean, sd)
  repeat { bad <- x < lo | x > hi; if (!any(bad)) break; x[bad] <- rnorm(sum(bad), mean, sd) }
  x
}

pairs <- read.csv("marburg_transmission_pairs.csv", stringsAsFactors = FALSE)
pairs <- pairs[pairs$confidence %in% c("high", "medium"), ]
pstat <- aggregate(serial_interval_days ~ outbreak, pairs,
                   function(v) c(mean = mean(v), sd = sd(v), n = length(v)))
pstat <- do.call(rbind, lapply(seq_len(nrow(pstat)), function(i)
  data.frame(outbreak = pstat$outbreak[i], mean = pstat$serial_interval_days[i, "mean"],
             sd = pstat$serial_interval_days[i, "sd"], n = pstat$serial_interval_days[i, "n"])))

gr <- read.csv("outbreak_growth_rates.csv", check.names = FALSE, stringsAsFactors = FALSE)
gr <- gr[!is.na(gr$growth_rate_per_day), ]

out <- data.frame()
for (i in seq_len(nrow(gr))) {
  label <- gr$outbreak[i]; rh <- gr$growth_rate_per_day[i]; se <- gr$std_error[i]
  key <- sub(" \\(excl. index\\)$", "", label)
  ps <- pstat[pstat$outbreak == key, ]
  if (nrow(ps) == 1 && ps$n >= MIN_PAIRS) {
    mu <- ps$mean; mse <- ps$sd / sqrt(ps$n); src <- sprintf("outbreak-pair mean (n=%d)", ps$n)
  } else { mu <- POOL_MEAN; mse <- POOL_SE; src <- "Qian pooled mean" }
  r  <- rnorm(N, rh, se)
  tg <- rtnorm(N, mu, mse, lo = 0.1, hi = 60)
  base <- 1 + r * tg / KAPPA
  R0 <- ifelse(base > 0, base^KAPPA, NA_real_)
  q <- quantile(R0, c(0.025, 0.5, 0.975), na.rm = TRUE)
  out <- rbind(out, data.frame(outbreak = label, gi_source = src,
    gi_mean_days = round(mu, 2), gi_cv_pooled = POOL_CV,
    r_per_day = rh, r_lo = round(rh - Z * se, 4), r_hi = round(rh + Z * se, 4),
    R0_gamma_median = round(q[[2]], 2), R0_gamma_lo = round(q[[1]], 2),
    R0_gamma_hi = round(q[[3]], 2), stringsAsFactors = FALSE))
}
write.csv(out, "outbreak_R0_montecarlo.csv", row.names = FALSE)
print(out)
