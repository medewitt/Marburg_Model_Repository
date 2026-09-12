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

```r
glm(cases ~ t, family = poisson())   # t = onset/report date - start of outbreak, in days
```

The slope on `t` is the intrinsic growth rate `r`, reported with a large-sample (Wald) 95% CI and doubling time `ln(2)/r`.
Outbreaks with peak daily incidence <= 1 or an ascending phase < 3 days are reported as NA (`estimate_growth_rates.R` / `.jl` -> `outbreak_growth_rates.csv`).

## Transmission pairs

Infector-infectee pairs were reconstructed from three source types, each pair tagged with a confidence tier (`high`/`medium`/`low`) and its basis (`marburg_transmission_pairs.csv`):

- **Two-case chains** from the repo `INTRO` flag (index vs secondary): Belgrade 1967, Kenya 1980.
- **Published case narratives**: the Tanzania 2023 tree from the case-by-case descriptions in Mmbaga et al. (index -> caregivers/HCWs, then mother -> grandchild), mapped to line-list onset dates.
- **Published transmission-tree figures**, digitized:
  - South Africa 1975: index -> companion and index -> nurse (Slenczka & Klenk, J Infect Dis 2007;196(S2):S131-5).
  - Uganda 2017 Kween: Figure 2 of Nyakarahuka et al., PLoS NTD 2019 (`10.1371/journal.pntd.0007257`).
  - Uganda 2012: Figure 5B of Knust et al., JID 2015 (`10.1093/infdis/jiv351`); Patient N was matched to the line-list ID by age and sex (1:1), and only solid (confirmed-contact) arrows were used, with exact onset dates from the line list. Dashed/tentative links and the tangled early Ibanda household were excluded.
  - DRC 1998-2000: Figure 2 of Bausch et al., NEJM 2006;355:909-919 (PMID 16943403), whose identical-sequence, epidemiologically-linked household pairs carry onset dates encoded in the sequence labels (e.g. `10DRC99aug06`).

The serial interval of each pair is `onset(infectee) - onset(infector)` in days; summaries and a method-of-moments gamma fit are in `estimate_serial_interval.R` / `.jl` -> `serial_interval_summary.csv`.
The pooled published estimate (Qian et al. 2023, medRxiv `10.1101/2022.06.17.22276538`; BMC Medicine `10.1186/s12916-023-03108-x`) is recorded in `literature_serial_intervals.csv` (gamma mean 9.2 d, SD 4.4).

## Reproduction number

Two pseudo-R0 estimates use the growth rate and the generation interval `Tg` (the serial interval as a proxy):

- **Linear** `R0 = 1 + r*Tg` (Wallinga & Lipsitch 2007, exponential generation interval) at fixed `Tg` = 9.3 and 11.2 d, in `outbreak_growth_rates.csv`.
- **Monte Carlo gamma** `R0 = (1 + r*Tg/kappa)^kappa`, `kappa = 1/CV^2`, propagating `r ~ Normal(r_hat, se)`, `Tg ~ Normal(mu, se_mu)`, with `mu` from that outbreak's own pairs when it has >= 3 (else the pooled 9.2 d) and the dispersion fixed at the pooled `CV = 4.4/9.2 = 0.478` (per-outbreak pair SDs are unreliable for star-shaped samples).
  N = 2e5, seed 1834 (`estimate_R0_montecarlo.R` / `.jl` -> `outbreak_R0_montecarlo.csv`).

## Figures

- `speed_strength_plane.tex` — the r-Tg (speed-strength) plane with gamma-GI R0 contours.
- `r_tg_timeline.tex` — `r` and the mean serial interval by outbreak year.

Both are pgfplots/TikZ; compile with `pdflatex` + `pgfplots`.

## Caveats

The serial interval is used as a proxy for the generation interval.
Poisson GLM fits assume mean = variance; sparse daily counts are likely overdispersed, so SEs are optimistic.
Tanzania 2023's `r` is inflated by a 9-day gap between the index case and the hospital cluster (a variant excluding the index is included).
DRC 1999's growth window is long, so its `r` is a whole-epidemic average.
The CSV values were generated with an equivalent Python implementation because R and Julia were unavailable in the working environment; the R scripts reproduce them.
