# Monte Carlo pseudo-R0 with an outbreak-informed generation interval.
#
# Growth rate is the NEGATIVE BINOMIAL estimate (r_nb, se_nb from
# outbreak_growth_rates.csv). The generation-interval mean is the DOUBLE
# INTERVAL-CENSORED gamma mean of that outbreak's high+medium transmission pairs
# when it has >= 3 (primarycensored, pwindow = swindow = 1; matches
# estimate_serial_interval.R), otherwise the pooled Qian et al. 2023 mean 9.2 d.
# GI dispersion is fixed at the pooled literature CV = 0.478 for all outbreaks:
# per-outbreak pair samples are star-shaped and their SD is not a reliable
# dispersion, so only the mean is taken from the pairs.
#
# R0 = (1 + r*T_g/kappa)^kappa, kappa = 1/CV^2 (gamma generation interval;
# Wallinga & Lipsitch 2007). Draws with base <= 0 (strong decline) are dropped.
# Seed 1834; N = 2e5. Propagates r ~ Normal(r_nb, se_nb) and
# T_g ~ Normal(mu, se_mu) truncated to (0.1, 60).

set.seed(1834)
requireNamespace("primarycensored")
N <- 200000L
POOL_MEAN <- 9.2; POOL_SE <- 0.7; POOL_CV <- round(4.4 / 9.2, 3)  # 0.478
MIN_PAIRS <- 3L; Z <- qnorm(0.975); KAPPA <- 1 / POOL_CV^2

rtnorm <- function(n, mean, sd, lo = -Inf, hi = Inf) {
  x <- rnorm(n, mean, sd)
  repeat { bad <- x < lo | x > hi; if (!any(bad)) break; x[bad] <- rnorm(sum(bad), mean, sd) }
  x
}

# censored gamma mean of a daily-resolution delay sample (matches SI script)
cens_gamma_mean <- function(x) {
  nll <- function(lp) {
    p <- exp(lp)
    d <- primarycensored::dprimarycensored(x, pgamma, shape = p[1], scale = p[2],
                                           pwindow = 1, swindow = 1)
    -sum(log(pmax(d, 1e-300)))
  }
  m <- mean(x); v <- var(x)
  o <- optim(log(c(m^2 / v, v / m)), nll, method = "Nelder-Mead",
             control = list(reltol = 1e-10, maxit = 5000))
  prod(exp(o$par))  # shape * scale
}

pairs <- read.csv("marburg_transmission_pairs.csv", stringsAsFactors = FALSE)
pairs$si <- as.integer(as.Date(pairs$infectee_onset) - as.Date(pairs$infector_onset))
pairs <- pairs[pairs$confidence %in% c("high", "medium"), ]

gr <- read.csv("outbreak_growth_rates.csv", check.names = FALSE, stringsAsFactors = FALSE)
gr <- gr[!is.na(gr$r_nb), ]

out <- data.frame()
for (i in seq_len(nrow(gr))) {
  label <- gr$outbreak[i]; rh <- gr$r_nb[i]; se <- gr$se_nb[i]
  key <- sub(" \\(excl. index\\)$", "", label)
  x <- pairs$si[pairs$outbreak == key]
  if (length(x) >= MIN_PAIRS) {
    mu <- cens_gamma_mean(x); mse <- sd(x) / sqrt(length(x))
    src <- sprintf("outbreak-pair censored mean (n=%d)", length(x))
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
