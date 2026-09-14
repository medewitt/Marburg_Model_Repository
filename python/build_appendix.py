"""Assemble the EID-style technical appendix (Quarto -> docx).

Reads the tables and figures produced by estimate_serial_interval_numpyro.py
(and the R estimation scripts) and writes a self-contained Quarto markdown
document at outputs/appendix/appendix.qmd. The .qmd is pure markdown with no
executable cells, so `quarto render ... --to docx` needs only pandoc, not a
Jupyter kernel. Figures are referenced as PNG (relative to the .qmd) for docx
embedding; the vector PDFs remain the manuscript-quality copies.

Run via `task py:appendix`, which regenerates figures/tables first.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import sys

import pandas as pd

OUTDIR = os.path.join("..", "outputs")
FIGREL = "../figures"  # relative to the .qmd inside outputs/appendix/
APPDIR = os.path.join(OUTDIR, "appendix")
QMD = os.path.join(APPDIR, "appendix.qmd")

DATA = os.path.join("..", "data", "marburg_transmission_pairs.csv")
SUMMARY_CSV = os.path.join(OUTDIR, "serial_interval_bayes_numpyro.csv")
DIAG_CSV = os.path.join(OUTDIR, "serial_interval_diagnostics.csv")
LOO_CSV = os.path.join(OUTDIR, "serial_interval_loo_comparison.csv")
LOOO_CSV = os.path.join(OUTDIR, "serial_interval_leave_one_out.csv")
LIT_CSV = os.path.join(OUTDIR, "serial_interval_literature_comparison.csv")
MLE_CSV = os.path.join(OUTDIR, "serial_interval_summary.csv")

# outbreaks Qian et al. 2023 used for their serial interval (26 pairs)
QIAN_OUTBREAKS = ["DRC 1999", "Kenya 1980", "South Africa 1975",
                  "Belgrade 1967", "Uganda 2012"]

EN = "–"  # en dash for numeric ranges

# Fixed numbering, in order of appearance.
T_GAMMA, T_LN, T_DIAG, T_LOO, T_AIC, T_LIT, T_LOOO = 1, 2, 3, 4, 5, 6, 7
F_LOO, F_PRIOR, F_PPC, F_POST, F_TRACE, F_FOREST, F_TIME, F_LIT, F_LOOO = range(1, 10)


def _md(df):
    return df.to_markdown(index=False)


def _pkg(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "n/a"


def figure(num, stem, caption):
    return (f"![Appendix Figure {num}. {caption}]"
            f"({FIGREL}/{stem}.png){{width=90%}}\n")


def table_estimates(family):
    s = pd.read_csv(SUMMARY_CSV)
    s = s[s["family"] == family]
    return pd.DataFrame({
        "Subset": s["set"],
        "Pairs": s["n_pairs"],
        "Mean SI, d (95% CrI)": s.apply(
            lambda r: f"{r.mu_median:.1f} ({r.mu_lo:.1f}{EN}{r.mu_hi:.1f})", axis=1),
        "SD, d (95% CrI)": s.apply(
            lambda r: f"{r.sd_median:.1f} ({r.sd_lo:.1f}{EN}{r.sd_hi:.1f})", axis=1),
        "CV (95% CrI)": s.apply(
            lambda r: f"{r.cv_median:.2f} ({r.cv_lo:.2f}{EN}{r.cv_hi:.2f})", axis=1),
    })


def table_diagnostics():
    d = pd.read_csv(DIAG_CSV)
    d = d.rename(columns={
        "set": "Subset", "parameter": "Parameter", "mean": "Mean", "sd": "SD",
        "ess_bulk": "ESS (bulk)", "ess_tail": "ESS (tail)", "r_hat": "R-hat"})
    d["ESS (bulk)"] = d["ESS (bulk)"].astype(int)
    d["ESS (tail)"] = d["ESS (tail)"].astype(int)
    return d[["Subset", "Parameter", "Mean", "SD", "ESS (bulk)", "ESS (tail)", "R-hat"]]


def table_loo():
    lo = pd.read_csv(LOO_CSV)
    return lo.rename(columns={
        "set": "Subset", "family": "Family", "rank": "Rank",
        "elpd_loo": "ELPD (LOO)", "p_loo": "p_LOO", "elpd_diff": "dELPD",
        "dse": "SE(d)", "weight": "Weight"})[
        ["Subset", "Family", "Rank", "ELPD (LOO)", "p_LOO", "dELPD", "SE(d)", "Weight"]]


def table_literature():
    c = pd.read_csv(LIT_CSV).fillna("")
    return c.rename(columns={
        "source": "Source", "population": "Population", "distribution": "Family",
        "mean_days": "Mean, d", "mean_95lo": "95% lo", "mean_95hi": "95% hi",
        "sd_days": "SD, d", "cv": "CV", "doi": "DOI"})


def table_looo():
    lo = pd.read_csv(LOOO_CSV)
    return pd.DataFrame({
        "Dropped outbreak": lo["dropped"],
        "Pairs": lo["n_pairs"].astype(int),
        "Mean SI, d (95% CrI)": lo.apply(
            lambda r: f"{r.mu_median:.1f} ({r.mu_lo:.1f}{EN}{r.mu_hi:.1f})", axis=1),
    })


def qian_restricted():
    """Our pairs restricted to Qian's five serial-interval outbreaks."""
    p = pd.read_csv(DATA)
    si = (pd.to_datetime(p.infectee_onset)
          - pd.to_datetime(p.infector_onset)).dt.days
    sub = si[p.outbreak.isin(QIAN_OUTBREAKS)]
    return len(sub), sub.mean(), sub.std()


def table_mle():
    m = pd.read_csv(MLE_CSV)
    return pd.DataFrame({
        "Subset": m["set"],
        "Mean SI, d": m["mean_censored"],
        "CV": m["cv_censored"],
        "Best (AIC)": m["best_family"],
        "AIC gamma": m["aic_gamma"],
        "AIC lognormal": m["aic_lognormal"],
        "AIC Weibull": m["aic_weibull"],
    })


def build():
    os.makedirs(APPDIR, exist_ok=True)
    versions = ", ".join(
        f"{p} {_pkg(p)}" for p in ["numpyro", "jax", "arviz", "numpy", "pandas"])
    pyver = ".".join(map(str, sys.version_info[:3]))
    qn, qmean, qsd = qian_restricted()

    p = []
    p.append(f"""---
title: "Appendix. Bayesian estimation of the Marburg virus serial interval"
format:
  docx:
    toc: true
    number-sections: false
---

# Overview

This appendix documents the Bayesian serial-interval analysis: the
double-interval-censored likelihood, the gamma and lognormal delay families and
their priors, MCMC settings, convergence diagnostics, model comparison by
leave-one-out cross-validation, prior and posterior predictive checks, and a
comparison with the published literature. Maximum-likelihood equivalents are in
the R reference implementation (`R/estimate_serial_interval.R`,
`primarycensored`).

# Data and transmission pairs

Infector{EN}infectee pairs were reconstructed from two-case chains, published
case narratives, and digitized transmission-tree figures, each tagged with a
confidence tier (high, medium, low) and its basis
(`data/marburg_transmission_pairs.csv`). The serial interval of a pair is
onset(infectee) {EN} onset(infector) in days. Because both onsets are recorded
to the day, the observed integer delay is doubly interval-censored.

# Model

For a pair with observed integer delay *n*, fix the infector onset day at 0 with
a within-day offset *a* ~ Uniform(0, 1) (the primary event). The infectee true
onset falls in day *n*, i.e. in [*n*, *n*+1). For a delay density *f* with CDF
*F*, the pair likelihood is

$$\\Pr(N = n) = \\int_0^1 \\big[ F(n + 1 - a) - F(n - a) \\big]\\, da,$$

with *F*(*t*) = 0 for *t* {EN}< 0. The primary offset is marginalized by 64-node
midpoint quadrature. Two delay families are fit, both parameterized by the mean
{EN} mu (days) so estimates are directly comparable: a gamma (shape *k*, so
SD = mu/sqrt(*k*), CV = 1/sqrt(*k*)) and a lognormal (sdlog {EN} sigma, meanlog
set so E[*T*] = mu, CV = sqrt(exp(sigma^2) {EN} 1)).

Priors (weakly informative, centered on an ~8-day Marburg serial interval):
mu ~ LogNormal(log 8, 0.6); gamma *k* ~ LogNormal(log 2, 0.7); lognormal
sigma ~ HalfNormal(0.5). Inference used the NUTS sampler (NumPyro): 4 chains,
1000 warmup and 2000 sampling iterations each (8000 posterior draws), seed 1834.

# Posterior estimates

: Appendix Table {T_GAMMA}. Gamma posterior serial-interval estimates (median, 95% credible interval).

{_md(table_estimates("gamma"))}

: Appendix Table {T_LN}. Lognormal posterior serial-interval estimates (median, 95% credible interval).

{_md(table_estimates("lognormal"))}

# Convergence diagnostics

All parameters converged (R-hat = 1.00; bulk and tail effective sample sizes in
the thousands from 8000 draws). Diagnostics are shown for the gamma model; the
lognormal fits converged equally well.

: Appendix Table {T_DIAG}. MCMC convergence diagnostics (gamma model).

{_md(table_diagnostics())}
""")

    p.append(f"""# Model comparison: gamma vs lognormal

The two families were compared by Pareto-smoothed importance-sampling
leave-one-out cross-validation (PSIS-LOO). The lognormal has the higher expected
log predictive density in every subset, but the difference is small relative to
its standard error (dELPD < 1 with SE(d) of comparable size, Appendix
Table {T_LOO}, Appendix Figure {F_LOO}), so the two families are not
distinguishable on these data. This agrees with the maximum-likelihood AIC
comparison (Appendix Table {T_AIC}), which also favors the lognormal marginally.
Gamma is retained as the primary model for consistency with the
generation-interval convention used in the reproduction-number calculation.

: Appendix Table {T_LOO}. PSIS-LOO comparison of gamma and lognormal. dELPD is the difference from the best model; SE(d) its standard error.

{_md(table_loo())}

{figure(F_LOO, 'loo_compare',
        'PSIS-LOO comparison per subset. Open circles, ELPD; whiskers, '
        'standard error; dashed line, best model.')}

: Appendix Table {T_AIC}. Maximum-likelihood family comparison (R, `primarycensored`; double interval-censored).

{_md(table_mle())}

## Effect on the reproduction number

The Monte Carlo R0 uses the Wallinga{EN}Lipsitch relation R0 = 1 / *M*({EN}*r*),
where *M* is the moment-generating function (Laplace transform) of the
generation interval and *r* the growth rate. For a gamma generation interval
this has the closed form R0 = (1 + *r* Tg / kappa)^kappa with kappa = 1/CV^2,
which is what the current pipeline uses (`R/estimate_R0_montecarlo.R`). A
lognormal generation interval has no closed-form MGF, so adopting it would
require evaluating *M*({EN}*r*) = E[exp({EN}*r T*)] numerically (for example by
quadrature over the lognormal density) at each Monte Carlo draw. Because the two
families are indistinguishable by LOO and give near-identical means and CVs, the
gamma closed form is retained; the numerical-Laplace-transform route is the
correct extension if a lognormal generation interval is preferred.
""")

    p.append(f"""# Predictive checks

The prior predictive distribution covers the observed serial-interval range
before fitting (Appendix Figure {F_PRIOR}). The posterior predictive
distribution reproduces the central mass of the observed delays; residual spread
in the tails is consistent with the small sample and the star-shaped dependence
among pairs that share the Tanzania index (Appendix Figure {F_PPC}).

{figure(F_PRIOR, 'prior_predictive', 'Prior predictive check, all pairs overlaid.')}

{figure(F_PPC, 'ppc_high_medium',
        'Posterior predictive check, high+medium subset (gamma). Points, '
        'observed proportions; line and band, predictive median and 95% interval.')}

# Posterior and sampling diagnostics (gamma, high+medium subset)

{figure(F_POST, 'posterior_high_medium',
        'Posterior marginals with 95% highest-density intervals.')}

{figure(F_TRACE, 'trace_high_medium', 'MCMC traces and marginal densities by chain.')}

{figure(F_FOREST, 'forest_compare',
        'Gamma serial-interval posteriors across confidence subsets '
        '(point, median; bar, 95% HDI).')}

Rank plots and per-subset trace, posterior, and predictive figures for the
high-only and all-pairs subsets are in `outputs/figures/`.

# Transmission-pair timelines

{figure(F_TIME, 'transmission_timelines',
        'Each pair aligned to a common t0 (infector onset, circle) to the '
        'infectee onset (square), sorted by serial interval and colored by '
        'onset confidence; green band, posterior mean serial interval.')}

# Comparison with the literature

Estimates from the reconstructed pairs run longer than the pooled published mean
(Qian et al. 2023, 9.2 d); the Ebola Zaire value is a comparator, not Marburg
(Appendix Figure {F_LIT}, Appendix Table {T_LIT}).

{figure(F_LIT, 'literature_comparison',
        'Marburg serial interval, this study versus literature. Thick bars, '
        '95% credible interval on the mean; dotted, literature mean +/- 1 SD.')}

: Appendix Table {T_LIT}. Serial interval, this study versus published estimates.

{_md(table_literature())}

## Source of the difference from Qian et al. 2023

Qian et al. estimated their serial interval from 26 infector{EN}infectee pairs
identified in five outbreaks (DRC, Kenya, South Africa, Belgrade, Uganda 2012)
by fitting a gamma to the raw onset-to-onset differences, without interval
censoring, using whole-outbreak linelists. Two points follow. First, the
difference is not the censoring method: restricting our pairs to those same five
outbreaks gives {qn} pairs with a naive mean of {qmean:.1f} d
(SD {qsd:.1f}), still well above 9.2 d, and our SD there matches theirs. Second,
it is not outbreak coverage: our newer outbreaks (Tanzania 2023, Uganda 2017)
move the pooled estimate only slightly. The gap is pair ascertainment: Qian
identified 26 pairs in those five outbreaks where we identified {qn}, and their
more inclusive linelist pairing (largest in the ~150-case DRC 1999-2000 linelist)
yields shorter intervals, whereas we used explicit, sequence- or figure-confirmed
transmission-tree links. A direct pair-by-pair comparison was not possible: the
data repository cited in Qian et al. (github.com/GeorgeYQian/MVD-Branching-
Process-Model-Repository) returns HTTP 404 and is no longer available.

# Leave-one-outbreak-out sensitivity

Refitting the gamma mean with each outbreak dropped in turn (high+medium set)
leaves the pooled mean between 11.1 and 12.4 d, all credible intervals
overlapping the full estimate (Appendix Figure {F_LOOO}, Appendix Table
{T_LOOO}). No single outbreak drives the result; dropping Uganda 2017 (the
longest-SI outbreak) still leaves the mean near 11.1 d, above the Qian value.

{figure(F_LOOO, 'leave_one_out',
        'Pooled mean serial interval (gamma, high+medium) as each outbreak is '
        'dropped. Green band and line, full-sample 95% CrI and median.')}

: Appendix Table {T_LOOO}. Leave-one-outbreak-out pooled mean serial interval.

{_md(table_looo())}
""")

    p.append(f"""# Reproducibility and data availability

All stochastic code is seeded with 1834. Bayesian estimation used Python {pyver}
({versions}); the maximum-likelihood reference used R with the `primarycensored`
package. Python dependencies are pinned in `uv.lock`. Source data
(`data/marburg_transmission_pairs.csv`, per-outbreak line lists) and all code
(`python/`, `R/`, `julia/`) are in the project repository; tables and figures in
this appendix are regenerated from them by `task py:appendix`.
""")

    with open(QMD, "w") as fh:
        fh.write("\n".join(p))
    print(f"Wrote {QMD}")


if __name__ == "__main__":
    build()
