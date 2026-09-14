# Marburg serial interval from reconstructed infector-infectee pairs, fit with a
# DOUBLE INTERVAL-CENSORED likelihood (the epidist / primarycensored model).
#
# Both onset dates are recorded to the day, so the observed integer delay is
# doubly interval-censored: the true infector and infectee onset times each lie
# in a one-day window. Treating the delay as an exact integer (as a naive
# method-of-moments gamma does) inflates the dispersion. We instead maximise the
# primary-event-censored likelihood implemented in the primarycensored package
# (Park et al.; the same censoring engine used by epidist), with a uniform
# within-day primary event and a one-day secondary censoring window:
#
#   Pr(N = n) = integral_0^1 integral_0^1 f(n + v - u) du dv
#             = dprimarycensored(n, pdist, pwindow = 1, swindow = 1)
#
# Gamma, lognormal and Weibull delay families are fit by maximum likelihood and
# compared by AIC. A nonparametric bootstrap (resampling pairs, seed 1834) gives
# a 95% CI on the mean. The naive (uncensored) SD/CV are reported alongside to
# show the censoring correction.
#
# Bayesian equivalent (not run here; produces posterior intervals):
#   library(epidist)
#   epidist(as_epidist_linelist_data(infector_onset, infector_onset + 1,
#           infectee_onset, infectee_onset + 1), family = "gamma", seed = 1834)
#
# CAVEATS: (1) serial interval != generation interval; (2) most high-tier pairs
# share the Tanzania index as infector (star-shaped), so their onset windows are
# not independent and the fitted SD is a lower bound on true dispersion.

set.seed(1834)
requireNamespace("primarycensored")  # CRAN: install.packages("primarycensored")

pairs <- read.csv("../data/marburg_transmission_pairs.csv", stringsAsFactors = FALSE)
pairs$si <- as.integer(as.Date(pairs$infectee_onset) - as.Date(pairs$infector_onset))

# negative log-likelihood of a double-interval-censored delay sample -----------
nll <- function(logpar, x, pdist) {
  p <- exp(logpar)
  args <- switch(attr(pdist, "fam"),
    gamma     = list(shape = p[1], scale = p[2]),
    lognormal = list(meanlog = log(p[1]), sdlog = p[2]),  # p[1]=exp(meanlog) for +ve start
    weibull   = list(shape = p[1], scale = p[2]))
  d <- do.call(primarycensored::dprimarycensored,
               c(list(x = x, pdist = pdist, pwindow = 1, swindow = 1), args))
  -sum(log(pmax(d, 1e-300)))
}

fam_defs <- list(
  gamma     = structure(pgamma,   fam = "gamma"),
  lognormal = structure(plnorm,   fam = "lognormal"),
  weibull   = structure(pweibull, fam = "weibull"))

moments <- function(fam, p) {
  switch(fam,
    gamma     = c(mean = p[1] * p[2], sd = sqrt(p[1]) * p[2]),
    lognormal = { m <- p[1]; s <- p[2]
                  c(mean = m * exp(s^2 / 2), sd = m * exp(s^2 / 2) * sqrt(exp(s^2) - 1)) },
    weibull   = c(mean = p[2] * gamma(1 + 1 / p[1]),
                  sd = p[2] * sqrt(gamma(1 + 2 / p[1]) - gamma(1 + 1 / p[1])^2)))
}

start <- function(fam, x) {
  m <- mean(x); v <- var(x); cv2 <- v / m^2
  switch(fam,
    gamma     = c(m^2 / v, v / m),
    lognormal = c(m / sqrt(1 + cv2), sqrt(log(1 + cv2))),
    weibull   = c(1.2, m / 0.9))
}

fit_family <- function(fam, x) {
  o <- optim(log(start(fam, x)), nll, x = x, pdist = fam_defs[[fam]],
             method = "Nelder-Mead", control = list(reltol = 1e-10, maxit = 5000))
  p <- exp(o$par); mo <- moments(fam, p)
  list(par = p, mean = mo[["mean"]], sd = mo[["sd"]], cv = mo[["sd"]] / mo[["mean"]],
       aic = 2 * length(p) + 2 * o$value)
}

boot_mean <- function(fam, x, B = 1000) {
  m <- replicate(B, {
    xb <- sample(x, length(x), replace = TRUE)
    tryCatch(fit_family(fam, xb)$mean, error = function(e) NA_real_)
  })
  quantile(m, c(0.025, 0.975), na.rm = TRUE)
}

sets <- list("high only"    = pairs$confidence == "high",
             "high+medium"  = pairs$confidence %in% c("high", "medium"),
             "all incl low" = rep(TRUE, nrow(pairs)))

out <- do.call(rbind, Map(function(nm, idx) {
  x <- pairs$si[idx]
  fits <- lapply(names(fam_defs), fit_family, x = x)
  names(fits) <- names(fam_defs)
  g <- fits[["gamma"]]
  ci <- boot_mean("gamma", x)
  best <- names(which.min(vapply(fits, `[[`, numeric(1), "aic")))
  nsd <- sd(x)
  data.frame(set = nm, n_pairs = length(x),
             mean_censored = round(g$mean, 2), sd_censored = round(g$sd, 2),
             cv_censored = round(g$cv, 3),
             mean_lo = round(ci[[1]], 2), mean_hi = round(ci[[2]], 2),
             gamma_shape = round(g$par[1], 3), gamma_scale = round(g$par[2], 3),
             sd_naive = round(nsd, 2), cv_naive = round(nsd / mean(x), 3),
             best_family = best,
             aic_gamma = round(fits[["gamma"]]$aic, 2),
             aic_lognormal = round(fits[["lognormal"]]$aic, 2),
             aic_weibull = round(fits[["weibull"]]$aic, 2),
             stringsAsFactors = FALSE)
}, names(sets), sets))
rownames(out) <- NULL
write.csv(out, "../outputs/serial_interval_summary.csv", row.names = FALSE)
print(out)
