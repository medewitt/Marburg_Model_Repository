# Empirical Marburg serial interval from reconstructed infector-infectee pairs.
#
# Reads marburg_transmission_pairs.csv (onset dates of infector and infectee for
# documented / epidemiologically linked transmission pairs) and summarises the
# serial interval (infectee onset - infector onset) by evidence tier. A gamma
# distribution is fit by method of moments (shape = mean^2/var, scale = var/mean).
#
# Pair provenance (see the link_basis column):
#   - Tanzania 2023: transmission tree documented case-by-case in Mmbaga et al.,
#     PLoS ONE 2024 (doi:10.1371/journal.pone.0309762), mapped to onset dates in
#     MarburgTanzania2023Line.csv.
#   - Belgrade 1967, Kenya 1980: single index -> single secondary (repo INTRO flag).
#   - South Africa 1975: index -> companion and index -> nurse; star topology
#     assumed from the INTRO flags (medium confidence).
#
# CAVEATS: (1) serial interval != generation interval; (2) six of the high-tier
# pairs share one infector (the Tanzania index), so the SI *variance* is
# underestimated (shared-exposure star) and the small CV should not be read as
# the generation-interval CV; the mean is the reliable quantity.

pairs <- read.csv("marburg_transmission_pairs.csv", stringsAsFactors = FALSE)
pairs$serial_interval_days <-
  as.integer(as.Date(pairs$infectee_onset) - as.Date(pairs$infector_onset))

gamma_mom <- function(x) {
  m <- mean(x); v <- var(x)
  c(shape = m^2 / v, scale = v / m)
}
summ <- function(x) {
  g <- gamma_mom(x)
  data.frame(n_pairs = length(x), mean = round(mean(x), 2), sd = round(sd(x), 2),
             cv = round(sd(x) / mean(x), 3), median = round(median(x), 1),
             gamma_shape = round(g[["shape"]], 2), gamma_scale = round(g[["scale"]], 2),
             gamma_mean = round(g[["shape"]] * g[["scale"]], 2))
}

sets <- list("high only"    = pairs$confidence == "high",
             "high+medium"  = pairs$confidence %in% c("high", "medium"),
             "all incl low" = rep(TRUE, nrow(pairs)))

out <- do.call(rbind, Map(function(nm, idx) cbind(set = nm, summ(pairs$serial_interval_days[idx])),
                          names(sets), sets))
rownames(out) <- NULL
write.csv(out, "serial_interval_summary.csv", row.names = FALSE)
print(out)
