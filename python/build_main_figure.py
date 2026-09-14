"""Main paper figure: is Marburg pathogen speed accelerating over time?

A 2x2 panel:
  A  Estimated serial-interval CDF (double interval-censored gamma, high+medium)
     with the naive empirical CDF and the Qian et al. 2023 pooled gamma, showing
     the censoring correction and the difference from the pooled published fit.
  B  Intrinsic growth rate r by outbreak versus year (95% CI): the speed measure.
  C  Mean serial interval Tg by outbreak versus year (mean +/- SE), with the
     pooled posterior mean band: the generation-interval measure.
  D  Speed-strength plane, r versus Tg with gamma-GI R0 contours, points colored
     by year: the synthesis. No drift toward the high-r corner over time is the
     "no acceleration" result; R appears as contours.

Inputs (all regenerated upstream by `task py:serial`, plus the curated
outbreak_r_tg_timeline.csv join of growth rates and per-outbreak serial
intervals). Run via `task py:mainfig`.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.stats import gamma as gamma_dist

import build_timeline  # regenerates outbreak_r_tg_timeline.csv from source

DATA = os.path.join("..", "data", "marburg_transmission_pairs.csv")
LIT = os.path.join("..", "data", "literature_serial_intervals.csv")
OUTDIR = os.path.join("..", "outputs")
FIGDIR = os.path.join(OUTDIR, "figures")
TIMELINE = os.path.join(OUTDIR, "outbreak_r_tg_timeline.csv")
DRAWS = os.path.join(OUTDIR, "serial_interval_posterior_draws.csv")
SUMMARY = os.path.join(OUTDIR, "serial_interval_bayes_numpyro.csv")

POOLED_CV = 0.478  # Qian pooled Marburg CV (4.4 / 9.2)
POOLED_TG = 9.2    # pooled mean generation interval (days)
KAPPA = 1.0 / POOLED_CV**2  # gamma-GI shape for R0 contours

OKABE = {"blue": "#0072B2", "orange": "#E69F00", "vermillion": "#D55E00",
         "green": "#009E73", "black": "#000000"}

# per-outbreak label offsets (dx pt, dy pt, ha, va), tuned to avoid overlaps
_DEFAULT_OFF = (4, 4, "left", "bottom")
LBL_R = {  # panel B (r vs year)
    "DRC 1999": (0, -11, "center", "top"),
    "Uganda 2012": (0, 9, "center", "bottom"),
    "Marburg 1967": (7, 0, "left", "center"),
    "Tanzania 2023": (-9, 0, "right", "center"),
    "Rwanda 2024": (-9, -4, "right", "top"),
}
LBL_TG = {  # panel C (Tg vs year)
    "South Africa 1975": (7, 0, "left", "center"),
    "DRC 1999": (0, -11, "center", "top"),
    "Uganda 2012": (7, 4, "left", "bottom"),
    "Uganda 2017": (0, -11, "center", "top"),
    "Tanzania 2023": (-9, 0, "right", "center"),
}
LBL_PLANE = {  # panel D (speed-strength)
    "DRC 1999": (-9, 0, "right", "center"),
    "Uganda 2012": (7, 5, "left", "bottom"),
    "Marburg 1967": (0, -11, "center", "top"),
    "Rwanda 2024": (8, -9, "left", "top"),
    "Tanzania 2023": (-9, -8, "right", "top"),
}


def _label(ax, x, y, text, offsets):
    dx, dy, ha, va = offsets.get(text.split("\n")[0], _DEFAULT_OFF)
    ax.annotate(text, (x, y), fontsize=6, xytext=(dx, dy),
                textcoords="offset points", ha=ha, va=va)


def panel_cdf(ax):
    draws = pd.read_csv(DRAWS)
    pairs = pd.read_csv(DATA)
    hm = pairs[pairs["confidence"].isin(["high", "medium"])]
    si = (pd.to_datetime(hm["infectee_onset"])
          - pd.to_datetime(hm["infector_onset"])).dt.days.to_numpy()

    t = np.linspace(0, 30, 300)
    # fitted interval-censored gamma CDF band from posterior draws
    scale = (draws["mu"] / draws["shape"]).to_numpy()[:, None]
    shape = draws["shape"].to_numpy()[:, None]
    cdf = gamma_dist.cdf(t[None, :], a=shape, scale=scale)
    lo, med, hi = np.percentile(cdf, [2.5, 50, 97.5], axis=0)
    ax.fill_between(t, lo, hi, color=OKABE["blue"], alpha=0.2)
    ax.plot(t, med, color=OKABE["blue"], lw=1.8,
            label="interval-censored gamma (this study)")

    # naive empirical CDF of the observed integer delays
    xs = np.sort(si)
    ax.step(np.concatenate([[0], xs]),
            np.concatenate([[0], np.arange(1, xs.size + 1) / xs.size]),
            where="post", color=OKABE["black"], lw=1.2, alpha=0.8,
            label="empirical (naive)")

    # Qian et al. 2023 pooled gamma
    q_shape = (9.2 / 4.4) ** 2
    q_scale = 4.4**2 / 9.2
    ax.plot(t, gamma_dist.cdf(t, a=q_shape, scale=q_scale),
            color=OKABE["vermillion"], lw=1.6, ls="--",
            label="Qian et al. 2023 (pooled)")

    ax.set_xlabel("serial interval (days)")
    ax.set_ylabel("cumulative probability")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("A  Serial-interval distribution", loc="left", fontweight="bold")


def panel_r_time(ax, tl):
    d = tl[tl["r_per_day"].notna()].copy()
    yerr = np.vstack([d["r_per_day"] - d["r_lo"], d["r_hi"] - d["r_per_day"]])
    ax.errorbar(d["year"], d["r_per_day"], yerr=yerr, fmt="o", color=OKABE["blue"],
                ecolor=OKABE["blue"], elinewidth=1, capsize=3, ms=6)
    for _, r in d.iterrows():
        _label(ax, r["year"], r["r_per_day"], r["outbreak"], LBL_R)
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("outbreak year")
    ax.set_ylabel("growth rate $r$ (day$^{-1}$)")
    ax.set_title("B  Speed over time", loc="left", fontweight="bold")


def panel_tg_time(ax, tl):
    d = tl[(tl["tg_n_pairs"] >= 2)].copy()
    se = d["tg_sd_days"] / np.sqrt(d["tg_n_pairs"])
    # pooled posterior mean band (high+medium)
    s = pd.read_csv(SUMMARY)
    hm = s[(s["set"] == "high+medium") & (s["family"] == "gamma")].iloc[0]
    ax.axhspan(hm["mu_lo"], hm["mu_hi"], color=OKABE["green"], alpha=0.12,
               label=f"pooled mean SI 95% CrI")
    ax.axhline(hm["mu_median"], color=OKABE["green"], lw=1.2, ls="--")
    ax.errorbar(d["year"], d["tg_mean_days"], yerr=1.96 * se, fmt="s",
                color=OKABE["orange"], ecolor=OKABE["orange"], elinewidth=1,
                capsize=3, ms=6)
    for _, r in d.iterrows():
        _label(ax, r["year"], r["tg_mean_days"],
               f"{r['outbreak']}\n(n={int(r['tg_n_pairs'])})", LBL_TG)
    ax.set_xlabel("outbreak year")
    ax.set_ylabel("mean serial interval $T_g$ (days)")
    ax.set_ylim(0, 20)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("C  Generation interval over time", loc="left", fontweight="bold")


def panel_plane(ax, tl):
    d = tl[tl["r_per_day"].notna()].copy()
    # own Tg where >= 2 pairs, otherwise the pooled generation interval
    d["tg"] = np.where(d["tg_n_pairs"].fillna(0) >= 2, d["tg_mean_days"], POOLED_TG)

    # gamma-GI R0 contours: Tg = kappa*(R0^(1/kappa) - 1) / r
    rr = np.linspace(0.005, 0.32, 200)
    for R0 in (1.5, 2, 3, 4):
        c = KAPPA * (R0 ** (1.0 / KAPPA) - 1.0)
        ax.plot(rr, c / rr, color="grey", lw=0.8, alpha=0.6)
        ax.annotate(f"$R_0$={R0}", (c / 13.2, 13.2), fontsize=6, color="grey")
    ax.axhline(POOLED_TG, color="grey", ls=":", lw=0.8)
    ax.axvline(0, color="grey", ls="--", lw=0.8)

    norm = Normalize(d["year"].min(), d["year"].max())
    sc = ax.scatter(d["r_per_day"], d["tg"], c=d["year"], cmap="viridis",
                    norm=norm, s=70, zorder=3, edgecolor="k", linewidth=0.4)
    xerr = np.vstack([d["r_per_day"] - d["r_lo"], d["r_hi"] - d["r_per_day"]])
    ax.errorbar(d["r_per_day"], d["tg"], xerr=xerr, fmt="none",
                ecolor="grey", elinewidth=0.8, capsize=2, zorder=2)
    for _, r in d.iterrows():
        _label(ax, r["r_per_day"], r["tg"], r["outbreak"], LBL_PLANE)
    cb = ax.figure.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("year", fontsize=8)
    ax.set_xlim(-0.04, 0.32)
    ax.set_ylim(0, 14)
    ax.set_xlabel("growth rate $r$ (day$^{-1}$)")
    ax.set_ylabel("mean generation interval $T_g$ (days)")
    ax.set_title("D  Speed-strength plane", loc="left", fontweight="bold")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    build_timeline.build()  # refresh the r/Tg timeline from source before plotting
    tl = pd.read_csv(TIMELINE)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    panel_cdf(axes[0, 0])
    panel_r_time(axes[0, 1], tl)
    panel_tg_time(axes[1, 0], tl)
    panel_plane(axes[1, 1], tl)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"main_figure.{ext}"),
                    bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {os.path.join(FIGDIR, 'main_figure.pdf')} (+ .png)")


if __name__ == "__main__":
    main()
