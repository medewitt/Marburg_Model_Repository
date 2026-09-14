# Methods

This note describes how the growth-rate, reproduction-number, and serial-interval quantities in this repository were extracted and estimated.
All stochastic code is seeded with 1834; the R scripts are the reference implementation and the Julia scripts are equivalents.

## Case data

Most outbreaks are stored as line lists keyed on symptom-onset date (`ONSET_DATE`; `DT_ONSET` for DRC 1999).
Angola 2005 is aggregate surveillance (reported date + new cases), with no line list.
Two line lists were added in this work:

- `MarburgTanzania2023Line.csv` — 9 cases extracted from the S1 dataset of Mmbaga et al., PLoS ONE 2024 (`10.1371/journal.pone.0309762`).
- `MarburgRwanda2024Line.csv` — 66 confirmed cases digitized from the epidemic curve in Figure 1 of Nsanzimana et al., NEJM 2025 (`10.1056/NEJMoa2415816`).
  Bar heights were read as vector rectangles from the PDF and calibrated against the axes; the daily onset counts were validated against the WHO week-39 total (26).

## Growth rate

For each outbreak a complete daily incidence series is built and zero-filled, then restricted to the ascending phase (day 0 through the last day at peak daily incidence).
Daily case counts are overdispersed relative to the Poisson mean = variance assumption, so the primary model is a negative binomial (NB2) log-linear fit,

```r
MASS::glm.nb(cases ~ t)   # Var = mu + alpha*mu^2, alpha = 1/theta; t in days
```

with a Poisson fit kept as a sensitivity analysis.
The slope on `t` is the intrinsic growth rate `r`, reported with a large-sample (Wald) 95% CI and doubling time `ln(2)/r`.
A likelihood-ratio test of Poisson vs NB (boundary `0.5 chi^2_1`) gives `overdisp_p`.
Overdispersion is significant only for Rwanda 2024 (`p = 0.048`; NB widens the CI on `r` from the Poisson 0.074-0.166 to 0.064-0.181) and is mild and non-significant for DRC 1999 and Marburg 1967; where the counts are too sparse to identify the NB dispersion (e.g. Uganda 2012) the NB columns fall back to the Poisson fit and the note records it.
Outbreaks with peak daily incidence <= 1 or an ascending phase < 3 days are reported as NA (`estimate_growth_rates.R` / `.jl` -> `outbreak_growth_rates.csv`).

## Transmission pairs

Infector-infectee pairs were reconstructed from three source types, each pair tagged with a confidence tier (`high`/`medium`/`low`) and its basis (`marburg_transmission_pairs.csv`):

- **Two-case chains** from the repo `INTRO` flag (index vs secondary): Belgrade 1967, Kenya 1980.
  Belgrade onset dates and case details are taken from Crozier & Kuhn 2020 (Microbiol Mol Biol Rev, "A Forgotten Episode of Marburg Virus Disease: Belgrade, Yugoslavia, 1967"): the 45-year-old veterinarian Ž.St. (onset 1 Sep 1967) infected his 44-year-old wife R.St. (onset 11 Sep 1967) via nursing/fomite contact, a serial interval of 10 days.
- **Published case narratives**: the Tanzania 2023 tree from the case-by-case descriptions in Mmbaga et al. (index -> caregivers/HCWs, then mother -> grandchild), mapped to line-list onset dates.
- **Published transmission-tree figures**, digitized:
  - South Africa 1975: the primary investigation (Conrad et al., Am J Trop Med Hyg 1978;27(6):1210-5) establishes a person-to-person chain index (patient 1, onset 12 Feb) -> companion (patient 2, onset 19 Feb) -> nurse (patient 3, onset 26 Feb), each a 7-day serial interval; only patient 1 had the arthropod exposure (Wankie), patient 2 nursed patient 1, and patient 3 sat up with patient 2 the night patient 1 died. This corrects an earlier index -> nurse assignment (a skipped generation) taken from the Slenczka & Klenk 2007 review.
  - Uganda 2017 Kween: Figure 2 of Nyakarahuka et al., PLoS NTD 2019 (`10.1371/journal.pntd.0007257`).
  - Uganda 2012: Figure 5B of Knust et al., JID 2015 (`10.1093/infdis/jiv351`); Patient N was matched to the line-list ID by age and sex (1:1), and only solid (confirmed-contact) arrows were used, with exact onset dates from the line list. Dashed/tentative links and the tangled early Ibanda household were excluded.
  - DRC 1998-2000: Figure 2 of Bausch et al., NEJM 2006;355:909-919 (PMID 16943403), whose identical-sequence, epidemiologically-linked household pairs carry onset dates encoded in the sequence labels (e.g. `10DRC99aug06`).

The serial interval of each pair is `onset(infectee) - onset(infector)` in days.
Because both onsets are recorded to the day, the observed integer delay is doubly interval-censored: the true infector and infectee onset times each lie in a one-day window, so the true delay is `n + (v - u)` with `u, v ~ U(0,1)` and a triangular censoring kernel `1 - |w|` on `(-1, 1)`.
Treating the delay as an exact integer (method of moments) inflates the dispersion, so the fit instead maximizes the primary-event-censored likelihood implemented in the `primarycensored` package (the same censoring engine used by `epidist`), `Pr(N = n) = dprimarycensored(n, pdist, pwindow = 1, swindow = 1)`.
Gamma, lognormal, and Weibull families are fit by maximum likelihood and compared by AIC; a nonparametric bootstrap (resampling pairs, seed 1834) gives a 95% CI on the mean (`estimate_serial_interval.R` / `.jl` -> `serial_interval_summary.csv`).
Censoring shrinks the coefficient of variation relative to the naive estimate (high+medium tier: gamma mean 11.81 d, censored CV 0.344 vs naive 0.370; lognormal fits marginally better by AIC across all tiers but gamma is retained for the generation-interval convention).
The pooled published estimate (Qian et al. 2023, medRxiv `10.1101/2022.06.17.22276538`; BMC Medicine `10.1186/s12916-023-03108-x`) is recorded in `literature_serial_intervals.csv` (gamma mean 9.2 d, SD 4.4).

## Reproduction number

A Monte Carlo pseudo-R0 uses the growth rate and the generation interval `Tg` (the serial interval as a proxy), with a gamma generation interval:
`R0 = (1 + r*Tg/kappa)^kappa`, `kappa = 1/CV^2`, propagating `r ~ Normal(r_nb, se_nb)` (the negative binomial growth rate) and `Tg ~ Normal(mu, se_mu)` truncated to (0.1, 60).
`mu` is the double interval-censored gamma mean of that outbreak's high+medium pairs when it has >= 3 (else the pooled 9.2 d); the dispersion is fixed at the pooled `CV = 4.4/9.2 = 0.478` (per-outbreak pair SDs are unreliable for star-shaped samples).
N = 2e5, seed 1834 (`estimate_R0_montecarlo.R` / `.jl` -> `outbreak_R0_montecarlo.csv`).
Under this model Rwanda 2024 gives R0 = 2.71 (95% 1.73-4.23) and Marburg 1967 R0 = 2.10 (0.85-4.55); the negative binomial growth rate widens both intervals relative to Poisson.
The linear `R0 = 1 + r*Tg` (Wallinga & Lipsitch 2007, exponential generation interval) columns of the earlier `outbreak_growth_rates.csv` are dropped in favor of this gamma-GI Monte Carlo estimate.

## Figures

- `speed_strength_plane.tex` — the r-Tg (speed-strength) plane with gamma-GI R0 contours.
- `r_tg_timeline.tex` — `r` and the mean serial interval by outbreak year.

Both are pgfplots/TikZ; compile with `pdflatex` + `pgfplots`.

## Caveats

The serial interval is used as a proxy for the generation interval, and no right-truncation adjustment is applied (the pairs come from concluded outbreaks and digitized transmission trees, not a real-time growing epidemic).
The double interval-censored likelihood treats pairs as independent, but most high-tier pairs share the Tanzania index as infector (star-shaped), so their onset windows are not independent and the fitted SD is a lower bound on the true dispersion.
The negative binomial observation model addresses Poisson overdispersion, but its dispersion is only weakly identified for the small, sparse outbreaks, where the fit falls back to Poisson.
Tanzania 2023's `r` is inflated by a 9-day gap between the index case and the hospital cluster (a variant excluding the index is included).
DRC 1999's growth window is long, so its `r` is a whole-epidemic average.
The CSV values were generated with an equivalent Python implementation of the same models (negative binomial GLM; the `primarycensored` double-censored likelihood) because R and Julia were unavailable in the working environment; the R (`MASS`, `primarycensored`) and Julia (`GLM`, hand-coded censored likelihood) scripts are the reference implementation and reproduce them.
