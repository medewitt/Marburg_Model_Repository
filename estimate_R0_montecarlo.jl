# Julia counterpart of estimate_R0_montecarlo.R.
# Monte Carlo pseudo R0 with an outbreak-informed generation interval:
# T_g mean from that outbreak's transmission pairs when it has >= 3 (else the
# pooled Qian 2023 mean 9.2 d), and a fixed pooled GI dispersion CV = 0.478
# (per-outbreak pair SDs are star-shaped and unreliable). Gamma GI form
# R0 = (1 + r*T_g/kappa)^kappa, kappa = 1/CV^2. Seed 1834. See the R script
# for the full rationale.

using CSV, DataFrames, Distributions, Statistics, Random
Random.seed!(1834)

const N = 200_000
const POOL_MEAN, POOL_SE = 9.2, 0.7
const POOL_CV = round(4.4/9.2, digits=3)
const KAPPA = 1/POOL_CV^2
const MIN_PAIRS = 3
const Z = quantile(Normal(), 0.975)

function rtnorm(n, mean, sd, lo, hi)
    x = rand(Normal(mean, sd), n)
    while true
        bad = findall(v -> v < lo || v > hi, x)
        isempty(bad) && break
        x[bad] = rand(Normal(mean, sd), length(bad))
    end
    x
end

pairs = CSV.read("marburg_transmission_pairs.csv", DataFrame)
pairs = pairs[in.(pairs.confidence, Ref(["high","medium"])), :]
pstat = combine(groupby(pairs, :outbreak),
    :serial_interval_days => mean => :mean,
    :serial_interval_days => std => :sd,
    :serial_interval_days => length => :n)

gr = CSV.read("outbreak_growth_rates.csv", DataFrame; normalizenames = false)
gr = gr[.!ismissing.(gr.growth_rate_per_day), :]

out = DataFrame()
for row in eachrow(gr)
    label = row.outbreak; rh = Float64(row.growth_rate_per_day); se = Float64(row.std_error)
    key = replace(label, " (excl. index)" => "")
    ps = pstat[pstat.outbreak .== key, :]
    if nrow(ps) == 1 && ps.n[1] >= MIN_PAIRS
        mu = ps.mean[1]; mse = ps.sd[1]/sqrt(ps.n[1]); src = "outbreak-pair mean (n=$(ps.n[1]))"
    else
        mu = POOL_MEAN; mse = POOL_SE; src = "Qian pooled mean"
    end
    r  = rand(Normal(rh, se), N)
    tg = rtnorm(N, mu, mse, 0.1, 60.0)
    base = 1 .+ r .* tg ./ KAPPA
    R0 = [b > 0 ? b^KAPPA : NaN for b in base]
    lo, med, hi = quantile(filter(!isnan, R0), [0.025, 0.5, 0.975])
    push!(out, (; outbreak = label, gi_source = src, gi_mean_days = round(mu, digits=2),
        gi_cv_pooled = POOL_CV, r_per_day = rh,
        r_lo = round(rh - Z*se, digits=4), r_hi = round(rh + Z*se, digits=4),
        R0_gamma_median = round(med, digits=2), R0_gamma_lo = round(lo, digits=2),
        R0_gamma_hi = round(hi, digits=2)))
end
CSV.write("outbreak_R0_montecarlo.csv", out)
show(out, allrows = true, allcols = true); println()
