# Julia counterpart of estimate_R0_montecarlo.R.
# Monte Carlo pseudo R0 propagating uncertainty in r, the mean generation
# interval T_g (era-specific: pre-2014 9.3 d, 2020s 11 d, with mean uncertainty),
# and the GI coefficient of variation, via the gamma-GI form
#   R0 = (1 + r*T_g/kappa)^kappa,  kappa = 1/CV^2.
# See estimate_R0_montecarlo.R for the full description. Seed 1834.
#
# Requires: CSV, DataFrames, Distributions, Statistics, Random.

using CSV, DataFrames, Distributions, Statistics, Random
Random.seed!(1834)

const N = 200_000
const SD_TG_MEAN = 1.0
const TG_MEAN = Dict("pre2014" => 9.3, "2020s" => 11.0)
const CV_MEAN, CV_SD, CV_LO, CV_HI = 0.6, 0.1, 0.3, 1.0
const Z = quantile(Normal(), 0.975)

# truncated-normal sampler (rejection)
function rtnorm(n, mean, sd, lo, hi)
    x = rand(Normal(mean, sd), n)
    while true
        bad = findall(v -> v < lo || v > hi, x)
        isempty(bad) && break
        x[bad] = rand(Normal(mean, sd), length(bad))
    end
    x
end

function era_of(label)
    m = match(r"(\d{4})", label)
    m === nothing && return missing
    parse(Int, m.captures[1]) < 2014 ? "pre2014" : "2020s"
end

gr = CSV.read("outbreak_growth_rates.csv", DataFrame; normalizenames = false)
gr = gr[.!ismissing.(gr.growth_rate_per_day), :]

rows = NamedTuple[]
for row in eachrow(gr)
    label = row.outbreak; rh = Float64(row.growth_rate_per_day); se = Float64(row.std_error)
    era = era_of(label); mu = TG_MEAN[era]
    r  = rand(Normal(rh, se), N)
    tg = rtnorm(N, mu, SD_TG_MEAN, 0.1, 50.0)
    cv = rtnorm(N, CV_MEAN, CV_SD, CV_LO, CV_HI)
    kappa = 1.0 ./ cv .^ 2
    base = 1.0 .+ r .* tg ./ kappa
    R0 = [b > 0 ? b^k : NaN for (b, k) in zip(base, kappa)]
    good = filter(!isnan, R0)
    lo, med, hi = quantile(good, [0.025, 0.5, 0.975])
    push!(rows, (; outbreak = label, era = era, r_per_day = rh, se_r = se,
        r_lo = round(rh - Z*se, digits=4), r_hi = round(rh + Z*se, digits=4),
        tg_mean_days = mu, sd_tg_mean = SD_TG_MEAN,
        tg_lo = round(mu - Z*SD_TG_MEAN, digits=2), tg_hi = round(mu + Z*SD_TG_MEAN, digits=2),
        cv_prior = "N($CV_MEAN,$CV_SD) trunc[$CV_LO,$CV_HI]",
        R0_gamma_median = round(med, digits=2), R0_gamma_lo = round(lo, digits=2),
        R0_gamma_hi = round(hi, digits=2),
        pct_undefined = round(count(isnan, R0) / N * 100, digits=1)))
end
res = DataFrame(rows)
CSV.write("outbreak_R0_montecarlo.csv", res)
show(res, allrows = true, allcols = true); println()
