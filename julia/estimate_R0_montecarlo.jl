# Julia counterpart of estimate_R0_montecarlo.R.
# Monte Carlo pseudo-R0 with an outbreak-informed generation interval. Growth
# rate is the negative binomial estimate (r_nb, se_nb); the T_g mean is the
# double interval-censored gamma mean of that outbreak's high+medium pairs when
# it has >= 3 (else the pooled Qian 2023 mean 9.2 d), and the GI dispersion CV is
# fixed at the pooled literature value 0.478. Gamma GI form
# R0 = (1 + r*T_g/kappa)^kappa, kappa = 1/CV^2. Seed 1834. See the R script for
# the full rationale.

using CSV, DataFrames, Distributions, Statistics, Random, Optim, FastGaussQuadrature, Dates
Random.seed!(1834)

const N = 200_000
const POOL_MEAN, POOL_SE = 9.2, 0.7
const POOL_CV = round(4.4/9.2, digits=3)
const KAPPA = 1/POOL_CV^2
const MIN_PAIRS = 3
const Z = quantile(Normal(), 0.975)

# censored gamma mean (triangular daily kernel; matches estimate_serial_interval.jl)
const GX, GW = gausslegendre(24)
_half(a, b) = (0.5*(b-a).*GX .+ 0.5*(a+b), 0.5*(b-a).*GW)
let (xa, wa) = _half(-1.0, 0.0), (xb, wb) = _half(0.0, 1.0)
    global WN = vcat(xa, xb)
    global WT = vcat(wa .* (1 .- abs.(xa)), wb .* (1 .- abs.(xb)))
end
function cens_gamma_mean(x)
    x = Float64.(x)
    nll(lp) = begin
        d = Gamma(exp(lp[1]), exp(lp[2]))
        -sum(log(max(sum(WT .* pdf.(d, n .+ WN)), 1e-300)) for n in x)
    end
    m = mean(x); v = var(x)
    o = optimize(nll, log.([m^2/v, v/m]), NelderMead(),
                 Optim.Options(g_tol = 1e-10, iterations = 5000))
    p = exp.(Optim.minimizer(o)); p[1]*p[2]
end

function rtnorm(n, mean, sd, lo, hi)
    x = rand(Normal(mean, sd), n)
    while true
        bad = findall(v -> v < lo || v > hi, x)
        isempty(bad) && break
        x[bad] = rand(Normal(mean, sd), length(bad))
    end
    x
end

pairs = CSV.read("../data/marburg_transmission_pairs.csv", DataFrame)
pairs = pairs[in.(pairs.confidence, Ref(["high","medium"])), :]
pairs.si = Dates.value.(Date.(pairs.infectee_onset) .- Date.(pairs.infector_onset))

gr = CSV.read("../outputs/outbreak_growth_rates.csv", DataFrame; normalizenames = false)
gr = gr[.!ismissing.(gr.r_nb), :]

out = DataFrame()
for row in eachrow(gr)
    label = row.outbreak; rh = Float64(row.r_nb); se = Float64(row.se_nb)
    key = replace(label, " (excl. index)" => "")
    x = pairs.si[pairs.outbreak .== key]
    if length(x) >= MIN_PAIRS
        mu = cens_gamma_mean(x); mse = std(x)/sqrt(length(x))
        src = "outbreak-pair censored mean (n=$(length(x)))"
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
CSV.write("../outputs/outbreak_R0_montecarlo.csv", out)
show(out, allrows = true, allcols = true); println()
