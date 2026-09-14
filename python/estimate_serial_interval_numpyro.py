"""Bayesian double-interval-censored Marburg serial interval (NumPyro + ArviZ).

The infector and infectee onset dates are each recorded to the day, so the
observed integer delay n is doubly interval-censored: the true onset times lie
in one-day windows. Treating n as an exact continuous delay inflates the
dispersion. This is the Bayesian counterpart of R/estimate_serial_interval.R,
which fits the same primary-event-censored likelihood by maximum likelihood via
the primarycensored package.

Likelihood. Fix the infector onset day at 0 with a within-day offset a ~ U(0, 1)
(the primary event). The infectee true onset falls in day n, i.e. in [n, n + 1).
For a delay density f with CDF F, the probability of the observed pair is

    Pr(N = n) = integral_0^1 [ F(n + 1 - a) - F(n - a) ] da,

with F(t) = 0 for t <= 0 (a positive delay distribution). The primary offset a
is marginalized by midpoint quadrature rather than sampled, which keeps the
parameter space low-dimensional and sampling clean.

Delay family. Gamma, parameterized by mean mu (days) and shape k so that
rate = k / mu, sd = mu / sqrt(k), cv = 1 / sqrt(k). Weakly informative priors
centered on a ~8-day Marburg serial interval.

Fits the same three confidence subsets as the R script and writes:
  outputs/serial_interval_bayes_numpyro.csv    posterior summaries (median, 95% CrI)
  outputs/serial_interval_diagnostics.csv      ESS (bulk, tail) and R-hat per parameter
  outputs/figures/trace_<subset>.pdf           MCMC trace + marginal density
  outputs/figures/posterior_<subset>.pdf       posterior marginals with 95% HDI
  outputs/figures/rank_<subset>.pdf            rank plots (convergence diagnostic)
  outputs/figures/forest_compare.pdf           posterior mean/sd/cv across subsets
  outputs/figures/transmission_timelines.pdf   infector->infectee onset timelines

CAVEATS mirror the R script: (1) serial interval != generation interval;
(2) most high-tier pairs share the Tanzania index as infector (star-shaped), so
their onset windows are not independent and the posterior sd is a lower bound on
true dispersion.
"""

from __future__ import annotations

import os

import numpyro

numpyro.set_host_device_count(4)  # before any JAX op, so 4 chains run in parallel

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import xarray as xr
import numpyro.distributions as dist
from jax.scipy.special import gammainc
from jax.scipy.stats import norm as jnorm
from numpyro.infer import MCMC, NUTS
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm as norm_dist

SEED = 1834
N_QUAD = 64  # midpoint nodes for the primary-offset integral over [0, 1]
FAMILIES = ["gamma", "lognormal"]
# per-family parameter list: shared mean/sd/cv plus the dispersion parameter
FAM_PARAMS = {"gamma": ["mu", "sd", "cv", "shape"],
              "lognormal": ["mu", "sd", "cv", "sigma"]}
PARAMS = FAM_PARAMS["gamma"]  # gamma is the primary model for diagnostic figures

DATA = os.path.join("..", "data", "marburg_transmission_pairs.csv")
LIT = os.path.join("..", "data", "literature_serial_intervals.csv")
OUTDIR = os.path.join("..", "outputs")
FIGDIR = os.path.join(OUTDIR, "figures")
SUMMARY_CSV = os.path.join(OUTDIR, "serial_interval_bayes_numpyro.csv")
DIAG_CSV = os.path.join(OUTDIR, "serial_interval_diagnostics.csv")
LIT_CSV = os.path.join(OUTDIR, "serial_interval_literature_comparison.csv")
LOO_CSV = os.path.join(OUTDIR, "serial_interval_loo_comparison.csv")
DRAWS_CSV = os.path.join(OUTDIR, "serial_interval_posterior_draws.csv")
LOOO_CSV = os.path.join(OUTDIR, "serial_interval_leave_one_out.csv")

# Okabe-Ito colorblind-safe palette
OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "skyblue": "#56B4E9",
    "purple": "#CC79A7",
}
CONF_COLOR = {"high": OKABE["blue"], "medium": OKABE["orange"], "low": OKABE["vermillion"]}

# midpoint quadrature nodes for a ~ Uniform(0, 1)
_A_NODES = (jnp.arange(N_QUAD) + 0.5) / N_QUAD


_SLUGS = {"high only": "high_only", "high+medium": "high_medium", "all incl low": "all"}


def _slug(label):
    return _SLUGS.get(label, label.replace(" ", "_").replace("+", "_"))


def _cdf_jax(t, family, mu, disp):
    """CDF at t (F(t) = 0 for t <= 0), gamma or lognormal, in JAX."""
    if family == "gamma":  # disp = shape
        return gammainc(disp, (disp / mu) * jnp.clip(t, min=0.0))
    # lognormal: disp = sdlog (sigma); meanlog set so E[T] = mu
    meanlog = jnp.log(mu) - disp**2 / 2.0
    tpos = jnp.clip(t, min=1e-12)
    return jnp.where(t > 0.0, jnorm.cdf((jnp.log(tpos) - meanlog) / disp), 0.0)


def model(n, family="gamma"):
    """Double-interval-censored delay likelihood for integer delays ``n``.

    ``family`` selects the delay distribution (gamma or lognormal), both
    parameterized by mean ``mu`` (days) so estimates are directly comparable.
    A per-observation log-likelihood is recorded for PSIS-LOO comparison.
    """
    mu = numpyro.sample("mu", dist.LogNormal(jnp.log(8.0), 0.6))
    if family == "gamma":
        disp = numpyro.sample("shape", dist.LogNormal(jnp.log(2.0), 0.7))
        numpyro.deterministic("sd", mu / jnp.sqrt(disp))
        numpyro.deterministic("cv", 1.0 / jnp.sqrt(disp))
    else:  # lognormal, disp = sdlog
        disp = numpyro.sample("sigma", dist.HalfNormal(0.5))
        cv = jnp.sqrt(jnp.expm1(disp**2))
        numpyro.deterministic("sd", mu * cv)
        numpyro.deterministic("cv", cv)

    # Pr(N = n) = mean_a [F(n + 1 - a) - F(n - a)], a over the quadrature grid.
    n_col = n[:, None]
    a = _A_NODES[None, :]
    cell = _cdf_jax(n_col + 1.0 - a, family, mu, disp) \
        - _cdf_jax(n_col - a, family, mu, disp)
    prob = jnp.clip(jnp.mean(cell, axis=1), min=1e-12)

    numpyro.deterministic("log_lik", jnp.log(prob))  # per-observation, for LOO
    numpyro.factor("loglik", jnp.sum(jnp.log(prob)))


def fit_subset(label, n, key, family):
    """Run NUTS for one family; return (summary row, InferenceData with log_lik)."""
    mcmc = MCMC(
        NUTS(lambda nn: model(nn, family=family)),
        num_warmup=1000,
        num_samples=2000,
        num_chains=4,
        progress_bar=False,
    )
    mcmc.run(key, jnp.asarray(n, dtype=jnp.float32))
    idata = az.from_numpyro(mcmc, log_likelihood=False)

    # move the per-observation log-lik deterministic into a log_likelihood group
    ll = idata.posterior["log_lik"]  # (chain, draw, log_lik_dim_0)
    idata.add_groups(log_likelihood=xr.Dataset(
        {"n_obs": ll.rename({ll.dims[-1]: "obs_id"})}))
    del idata.posterior["log_lik"]

    row = {"set": label, "family": family, "n_pairs": int(n.shape[0])}
    for par in FAM_PARAMS[family]:
        x = idata.posterior[par].values.reshape(-1)
        lo, med, hi = np.quantile(x, [0.025, 0.5, 0.975])
        row[f"{par}_median"] = med
        row[f"{par}_lo"] = lo
        row[f"{par}_hi"] = hi
    return row, idata


def leave_one_outbreak_out(si, outbreaks, key):
    """Refit the gamma mean SI dropping each outbreak in turn (high+medium set).

    Guards against any single outbreak driving the pooled estimate. Returns a
    DataFrame with the full fit and each leave-one-out fit.
    """
    rows = []
    key, sub = jax.random.split(key)
    full, _ = fit_subset("full (high+medium)", si.astype(float), sub, "gamma")
    rows.append({"dropped": "none (full)", "n_pairs": full["n_pairs"],
                 "mu_median": full["mu_median"], "mu_lo": full["mu_lo"],
                 "mu_hi": full["mu_hi"]})
    for ob in sorted(np.unique(outbreaks)):
        keep = outbreaks != ob
        key, sub = jax.random.split(key)
        r, _ = fit_subset(f"drop {ob}", si[keep].astype(float), sub, "gamma")
        rows.append({"dropped": ob, "n_pairs": r["n_pairs"],
                     "mu_median": r["mu_median"], "mu_lo": r["mu_lo"],
                     "mu_hi": r["mu_hi"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def save_fig(fig, stem):
    """Save PDF (vector, manuscript) and PNG (raster, docx appendix)."""
    fig.savefig(os.path.join(FIGDIR, f"{stem}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, f"{stem}.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_trace(idata, label):
    az.plot_trace(idata, var_names=PARAMS, compact=False, figsize=(9, 7.5))
    fig = plt.gcf()
    fig.suptitle(f"MCMC traces - {label}", y=1.01)
    fig.tight_layout()
    save_fig(fig, f"trace_{_slug(label)}")


def fig_posterior(idata, label):
    axes = az.plot_posterior(
        idata,
        var_names=PARAMS,
        hdi_prob=0.95,
        point_estimate="median",
        color=OKABE["blue"],
        figsize=(9, 6),
    )
    fig = np.ravel(axes)[0].get_figure()
    axmap = dict(zip(PARAMS, np.ravel(axes)))
    axmap["mu"].set_xlabel("mean serial interval (days)")
    axmap["sd"].set_xlabel("sd (days)")
    axmap["cv"].set_xlabel("coefficient of variation")
    axmap["shape"].set_xlabel("gamma shape")
    fig.suptitle(f"Posterior marginals (median, 95% HDI) - {label}", y=1.02)
    fig.tight_layout()
    save_fig(fig, f"posterior_{_slug(label)}")


def fig_rank(idata, label):
    az.plot_rank(idata, var_names=PARAMS, figsize=(9, 6))
    fig = plt.gcf()
    fig.suptitle(f"Rank plots (chain mixing) - {label}", y=1.01)
    fig.tight_layout()
    save_fig(fig, f"rank_{_slug(label)}")


def fig_forest(idatas, labels):
    # one panel per parameter so each keeps its own x-scale (cv ~0.4 vs mu ~12)
    panels = [("mu", "mean SI (days)"), ("sd", "sd (days)"),
              ("cv", "coefficient of variation")]
    colors = [OKABE["blue"], OKABE["orange"], OKABE["vermillion"]]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (par, xlab) in zip(axes, panels):
        az.plot_forest(
            idatas,
            model_names=labels,
            var_names=[par],
            combined=True,
            hdi_prob=0.95,
            colors=colors,
            ax=ax,
        )
        ax.set_xlabel(xlab)
        leg = ax.get_legend()  # drop per-panel legends; one figure legend instead
        if leg is not None:
            leg.remove()
    handles = [Line2D([0], [0], color=c, lw=2.5) for c in colors]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Serial interval posteriors across confidence subsets "
                 "(point = median, bar = 95% HDI)", y=1.02)
    fig.tight_layout()
    save_fig(fig, "forest_compare")


def fig_timelines(pairs, ref_lo, ref_med, ref_hi):
    """Infector -> infectee onset, each pair aligned to a common t0 = infector onset."""
    df = pairs.copy()
    df["si"] = (
        pd.to_datetime(df["infectee_onset"]) - pd.to_datetime(df["infector_onset"])
    ).dt.days
    df = df.sort_values("si").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 9))
    # posterior reference band (high+medium mean serial interval)
    ax.axvspan(ref_lo, ref_hi, color=OKABE["green"], alpha=0.12,
               label=f"posterior mean SI 95% CrI [{ref_lo:.1f}, {ref_hi:.1f}]")
    ax.axvline(ref_med, color=OKABE["green"], lw=1.4, ls="--",
               label=f"posterior mean SI = {ref_med:.1f} d")

    seen = set()
    for i, r in df.iterrows():
        c = CONF_COLOR.get(r["confidence"], "grey")
        ax.plot([0, r["si"]], [i, i], color=c, lw=1.5, zorder=2)
        lbl = r["confidence"] if r["confidence"] not in seen else None
        seen.add(r["confidence"])
        ax.scatter(0, i, color=c, marker="o", s=28, zorder=3)  # infector onset
        ax.scatter(r["si"], i, color=c, marker="s", s=28, zorder=3, label=lbl)  # infectee

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{r.infector_id}→{r.infectee_id}" for r in df.itertuples()],
                       fontsize=7)
    ax.set_ylim(-1, len(df))
    ax.invert_yaxis()

    xmax = int(np.ceil(df["si"].max() / 2) * 2)
    ticks = list(range(0, xmax + 1, 2))
    ax.set_xticks(ticks)
    ax.set_xticklabels(["t0"] + [f"t+{t}" for t in ticks[1:]])
    ax.set_xlabel("Days since infector symptom onset (common time)")
    ax.set_title("Marburg transmission pairs: infector (circle) → infectee (square)")
    ax.grid(axis="x", ls=":", alpha=0.5)

    # legend: confidence markers plus posterior band
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles, lbls, loc="upper right", fontsize=8, framealpha=0.9,
              title="onset confidence / model")
    fig.tight_layout()
    save_fig(fig, "transmission_timelines")


# --------------------------------------------------------------------------
# predictive checks
# --------------------------------------------------------------------------
def _pmf_family(mu, disp, family, ns):
    """Double-interval-censored PMF over integer grid ns, per (mu, disp) draw.

    Returns array of shape (n_draws, len(ns)); reuses the model likelihood so the
    predictive check is of exactly the fitted censored distribution.
    """
    mu = np.asarray(mu)[:, None, None]
    disp = np.asarray(disp)[:, None, None]
    a = ((np.arange(N_QUAD) + 0.5) / N_QUAD)[None, None, :]
    n = np.asarray(ns)[None, :, None]
    hi = np.clip(n + 1.0 - a, 0, None)
    lo = np.clip(n - a, 0, None)
    if family == "gamma":  # disp = shape
        scale = mu / disp
        F = lambda t: gamma_dist.cdf(t, a=disp, scale=scale)
    else:  # lognormal, disp = sdlog
        meanlog = np.log(mu) - disp**2 / 2.0
        F = lambda t: np.where(t > 0, norm_dist.cdf(
            (np.log(np.clip(t, 1e-12, None)) - meanlog) / disp), 0.0)
    return (F(hi) - F(lo)).mean(axis=2)


def fig_ppc(mu_s, disp_s, obs, title, stem, family="gamma", color=OKABE["blue"]):
    obs = np.asarray(obs, dtype=int)
    n_max = int(obs.max()) + 6
    ns = np.arange(0, n_max + 1)
    pmf = _pmf_family(mu_s, disp_s, family, ns)
    lo, med, hi = np.percentile(pmf, [2.5, 50, 97.5], axis=0)
    obs_p = np.bincount(obs, minlength=n_max + 1)[: n_max + 1] / obs.size

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(ns, lo, hi, color=color, alpha=0.2, label="predictive 95% interval")
    ax.plot(ns, med, color=color, lw=1.8, label="predictive median")
    ax.scatter(ns, obs_p, color=OKABE["vermillion"], zorder=3, s=28,
               label="observed proportion")
    ax.set_xlabel("serial interval (days, integer)")
    ax.set_ylabel("probability")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, stem)


def fig_prior_predictive(obs_all):
    rng = np.random.default_rng(SEED)
    mu = np.exp(np.log(8.0) + 0.6 * rng.standard_normal(2000))
    shape = np.exp(np.log(2.0) + 0.7 * rng.standard_normal(2000))
    fig_ppc(mu, shape, obs_all, "Prior predictive check (all pairs overlaid)",
            "prior_predictive", family="gamma", color=OKABE["green"])


def fig_loo(compares):
    """One PSIS-LOO comparison panel per subset (gamma vs lognormal)."""
    n = len(compares)
    fig, axes = plt.subplots(n, 1, figsize=(7, 2.6 * n), squeeze=False)
    for ax, (subset, cmp) in zip(axes[:, 0], compares.items()):
        az.plot_compare(cmp, ax=ax)
        ax.set_title(f"PSIS-LOO comparison - {subset}")
    fig.tight_layout()
    save_fig(fig, "loo_compare")


def fig_looo(df):
    """Caterpillar of the pooled mean SI as each outbreak is dropped."""
    full = df[df["dropped"] == "none (full)"].iloc[0]
    d = df[df["dropped"] != "none (full)"].iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axvspan(full["mu_lo"], full["mu_hi"], color=OKABE["green"], alpha=0.12,
               label="full 95% CrI")
    ax.axvline(full["mu_median"], color=OKABE["green"], lw=1.3, ls="--",
               label=f"full mean = {full['mu_median']:.1f} d")
    y = np.arange(len(d))
    xerr = np.vstack([d["mu_median"] - d["mu_lo"], d["mu_hi"] - d["mu_median"]])
    ax.errorbar(d["mu_median"], y, xerr=xerr, fmt="o", color=OKABE["blue"],
                ecolor=OKABE["blue"], capsize=3, ms=6)
    ax.set_yticks(y)
    ax.set_yticklabels([f"drop {r.dropped}" for r in d.itertuples()], fontsize=8)
    ax.set_xlabel("pooled mean serial interval (days)")
    ax.set_title("Leave-one-outbreak-out (gamma, high+medium)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", ls=":", alpha=0.5)
    fig.tight_layout()
    save_fig(fig, "leave_one_out")


def fig_literature(summary_rows, labels):
    """Compare posterior mean SI (this study) with published Marburg estimates."""
    lit = pd.read_csv(LIT)
    palette = [OKABE["blue"], OKABE["orange"], OKABE["vermillion"]]
    rows = []
    for i, lbl in enumerate(labels):
        r = summary_rows[i]
        rows.append(dict(name=f"This study: {lbl}", mean=r["mu_median"],
                         lo=r["mu_lo"], hi=r["mu_hi"], kind="cri", color=palette[i]))
    for _, lr in lit.iterrows():
        marburg = "Marburg" in lr["population"] and "NOT" not in lr["population"]
        pop = "Marburg, pooled" if marburg else "Ebola, comparator"
        rows.append(dict(name=f"{lr['source'].split(',')[0]} ({pop})",
                         mean=lr["mean_days"], lo=lr["mean_days"] - lr["sd_days"],
                         hi=lr["mean_days"] + lr["sd_days"], kind="sd",
                         color=OKABE["skyblue"] if marburg else "grey"))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for j, rw in enumerate(rows):
        yy = len(rows) - 1 - j
        if rw["kind"] == "cri":
            ax.plot([rw["lo"], rw["hi"]], [yy, yy], color=rw["color"], lw=3,
                    solid_capstyle="round")
        else:
            ax.plot([rw["lo"], rw["hi"]], [yy, yy], color=rw["color"], lw=1.2, ls=":")
        ax.scatter(rw["mean"], yy, color=rw["color"], s=45, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["name"] for r in reversed(rows)])
    ax.set_xlabel("mean serial interval (days)")
    ax.set_title("Marburg serial interval: this study vs literature")
    handles = [
        Line2D([0], [0], color="black", lw=3, label="this study (95% CrI on the mean)"),
        Line2D([0], [0], color="grey", lw=1.2, ls=":", label="literature (mean ±1 SD)"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.grid(axis="x", ls=":", alpha=0.5)
    fig.tight_layout()
    save_fig(fig, "literature_comparison")

    # tidy comparison table
    comp = []
    for i, lbl in enumerate(labels):
        r = summary_rows[i]
        comp.append(dict(source="This study (NumPyro, double-censored)",
                         population=f"Marburg, {lbl}", distribution="gamma",
                         mean_days=round(r["mu_median"], 2),
                         mean_95lo=round(r["mu_lo"], 2), mean_95hi=round(r["mu_hi"], 2),
                         sd_days="", cv="", doi=""))
    for _, lr in lit.iterrows():
        comp.append(dict(source=lr["source"], population=lr["population"],
                         distribution=lr["distribution"], mean_days=lr["mean_days"],
                         mean_95lo="", mean_95hi="", sd_days=lr["sd_days"],
                         cv=lr["cv"], doi=lr["doi"]))
    pd.DataFrame(comp).to_csv(LIT_CSV, index=False)


# --------------------------------------------------------------------------
def main():
    os.makedirs(FIGDIR, exist_ok=True)
    pairs = pd.read_csv(DATA)
    si = (
        pd.to_datetime(pairs["infectee_onset"])
        - pd.to_datetime(pairs["infector_onset"])
    ).dt.days.to_numpy()

    conf = pairs["confidence"].to_numpy()
    subsets = {
        "high only": conf == "high",
        "high+medium": np.isin(conf, ["high", "medium"]),
        "all incl low": np.ones(len(pairs), dtype=bool),
    }

    key = jax.random.PRNGKey(SEED)
    summary_rows, ln_rows, diag_rows, loo_rows = [], [], [], []
    gamma_idatas, labels, compares = [], [], {}
    for label, idx in subsets.items():
        obs = si[idx].astype(float)
        fits = {}
        for family in FAMILIES:
            key, sub = jax.random.split(key)
            row, idata = fit_subset(label, obs, sub, family)
            fits[family] = idata
            (summary_rows if family == "gamma" else ln_rows).append(row)

        gamma = fits["gamma"]
        gamma_idatas.append(gamma)
        labels.append(label)

        # gamma is the primary model: diagnostics table and diagnostic figures
        diag = az.summary(gamma, var_names=PARAMS, kind="all", hdi_prob=0.95)
        for par in PARAMS:
            diag_rows.append({
                "set": label, "parameter": par,
                "mean": diag.loc[par, "mean"], "sd": diag.loc[par, "sd"],
                "ess_bulk": diag.loc[par, "ess_bulk"],
                "ess_tail": diag.loc[par, "ess_tail"],
                "r_hat": diag.loc[par, "r_hat"],
            })
        fig_trace(gamma, label)
        fig_posterior(gamma, label)
        fig_rank(gamma, label)

        mu_s = gamma.posterior["mu"].values.reshape(-1)
        shape_s = gamma.posterior["shape"].values.reshape(-1)
        pick = np.random.default_rng(SEED).choice(mu_s.size, 1000, replace=False)
        fig_ppc(mu_s[pick], shape_s[pick], si[idx],
                f"Posterior predictive check - {label}", f"ppc_{_slug(label)}")

        # PSIS-LOO comparison of gamma vs lognormal for this subset
        cmp = az.compare({"gamma": fits["gamma"], "lognormal": fits["lognormal"]},
                         ic="loo")
        compares[label] = cmp
        for fam, r in cmp.iterrows():
            loo_rows.append({
                "set": label, "family": fam, "rank": int(r["rank"]),
                "elpd_loo": r["elpd_loo"], "p_loo": r["p_loo"],
                "elpd_diff": r["elpd_diff"], "dse": r["dse"],
                "weight": r["weight"], "se": r["se"],
            })

        print(f"\n=== {label} (n = {int(obs.size)}) ===")
        print(diag[["mean", "sd", "hdi_2.5%", "hdi_97.5%", "ess_bulk",
                    "ess_tail", "r_hat"]].to_string())
        print(cmp[["rank", "elpd_loo", "p_loo", "elpd_diff", "dse", "weight"]].to_string())

    fig_prior_predictive(si)
    fig_forest(gamma_idatas, labels)
    fig_loo(compares)
    fig_literature(summary_rows, labels)

    # leave-one-outbreak-out sensitivity (high+medium)
    hm_mask = np.isin(conf, ["high", "medium"])
    key, sub = jax.random.split(key)
    looo = leave_one_outbreak_out(si[hm_mask], pairs["outbreak"].to_numpy()[hm_mask], sub)
    fig_looo(looo)
    looo.round(2).to_csv(LOOO_CSV, index=False)

    # timeline uses the high+medium posterior mean SI as the reference band
    hm = summary_rows[labels.index("high+medium")]
    fig_timelines(pairs, hm["mu_lo"], hm["mu_median"], hm["mu_hi"])

    # save high+medium gamma posterior draws for the main-figure CDF band
    hm_g = gamma_idatas[labels.index("high+medium")]
    pd.DataFrame({"mu": hm_g.posterior["mu"].values.reshape(-1),
                  "shape": hm_g.posterior["shape"].values.reshape(-1)}) \
        .sample(2000, random_state=SEED).to_csv(DRAWS_CSV, index=False)

    summ = pd.DataFrame(summary_rows + ln_rows).round(3)
    summ.to_csv(SUMMARY_CSV, index=False)
    diag = pd.DataFrame(diag_rows).round(
        {"mean": 3, "sd": 3, "ess_bulk": 0, "ess_tail": 0, "r_hat": 3})
    diag.to_csv(DIAG_CSV, index=False)
    loo = pd.DataFrame(loo_rows).round(
        {"elpd_loo": 2, "p_loo": 2, "elpd_diff": 2, "dse": 2, "weight": 3, "se": 2})
    loo.to_csv(LOO_CSV, index=False)

    print(f"\nWrote {SUMMARY_CSV}")
    print(f"Wrote {DIAG_CSV}")
    print(f"Wrote {LOO_CSV}")
    print(f"Wrote {LOOO_CSV}")
    print(f"Wrote {LIT_CSV}")
    print("\nLeave-one-outbreak-out (pooled mean SI):")
    print(looo.round(2).to_string(index=False))
    print(f"Wrote figures (pdf + png) to {FIGDIR}/")
    print("\nLOO comparison (gamma vs lognormal):")
    print(loo.to_string(index=False))


if __name__ == "__main__":
    main()
