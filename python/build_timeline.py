"""Regenerate outbreak_r_tg_timeline.csv from source, replacing the hand-kept file.

Joins two products so a data edit propagates automatically:
  r_per_day, r_lo, r_hi  <- negative-binomial growth-rate fit
                            (outbreak_growth_rates.csv; R/estimate_growth_rates.R),
                            excluding the "(excl. index)" sensitivity variant.
  tg_mean_days, tg_n_pairs, tg_sd_days
                         <- naive per-outbreak serial interval over the
                            high+medium transmission pairs (the primary set).

Rows are the union of outbreaks that have a growth rate or at least one pair.
Consumed by build_main_figure.py (panels C and D).
"""

from __future__ import annotations

import os
import re

import pandas as pd

OUTDIR = os.path.join("..", "outputs")
PAIRS = os.path.join("..", "data", "marburg_transmission_pairs.csv")
GROWTH = os.path.join(OUTDIR, "outbreak_growth_rates.csv")
OUT = os.path.join(OUTDIR, "outbreak_r_tg_timeline.csv")


def _year(name):
    m = re.search(r"(?:19|20)\d{2}", name)
    return int(m.group()) if m else pd.NA


def build():
    g = pd.read_csv(GROWTH)
    g = g[g["r_nb"].notna() & ~g["outbreak"].str.contains(r"\(excl", regex=True)]
    r = g[["outbreak", "r_nb", "ci_low_nb", "ci_high_nb"]].rename(
        columns={"r_nb": "r_per_day", "ci_low_nb": "r_lo", "ci_high_nb": "r_hi"})

    p = pd.read_csv(PAIRS)
    p = p[p["confidence"].isin(["high", "medium"])].copy()
    p["si"] = (pd.to_datetime(p.infectee_onset)
               - pd.to_datetime(p.infector_onset)).dt.days
    tg = (p.groupby("outbreak")["si"]
          .agg(tg_mean_days="mean", tg_n_pairs="count", tg_sd_days="std")
          .reset_index())

    m = pd.merge(r, tg, on="outbreak", how="outer")
    m["year"] = m["outbreak"].map(_year)
    m["tg_n_pairs"] = m["tg_n_pairs"].fillna(0).astype(int)
    for c, nd in [("r_per_day", 4), ("r_lo", 4), ("r_hi", 4),
                  ("tg_mean_days", 2), ("tg_sd_days", 2)]:
        m[c] = m[c].round(nd)
    m = m.sort_values("outbreak")[
        ["outbreak", "year", "r_per_day", "r_lo", "r_hi",
         "tg_mean_days", "tg_n_pairs", "tg_sd_days"]]
    m.to_csv(OUT, index=False)
    return m


if __name__ == "__main__":
    out = build()
    print(f"Wrote {OUT} ({len(out)} outbreaks)")
    print(out.to_string(index=False))
