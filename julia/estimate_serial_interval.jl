# Julia counterpart of estimate_serial_interval.R.
# Marburg serial interval from reconstructed infector-infectee pairs, fit with a
# DOUBLE INTERVAL-CENSORED likelihood (the epidist / primarycensored model).
#
# Both onsets are daily, so the true delay tau = n + (v - u) with u,v ~ U(0,1),
# giving a triangular censoring kernel h(w) = 1 - |w| on (-1, 1):
#   Pr(N = n) = integral_{-1}^{1} (1 - |w|) f(n + w) dw
# (identical to primarycensored::dprimarycensored with pwindow = swindow = 1).
# Gamma, lognormal and Weibull are fit by MLE and compared by AIC; a bootstrap
# (seed 1834) gives a 95% CI on the mean. See estimate_serial_interval.R for
# provenance and caveats (SI != GI; star-shaped high-tier pairs underestimate
# the dispersion).

using CSV, DataFrames, Dates, Statistics, Distributions, Optim, FastGaussQuadrature, Random

Random.seed!(1834)

# fixed Gauss-Legendre nodes on [-1,0] and [0,1] (kernel has a corner at 0)
const GX, GW = gausslegendre(24)
half(a, b) = (0.5*(b-a).*GX .+ 0.5*(a+b), 0.5*(b-a).*GW)
let (xa, wa) = half(-1.0, 0.0), (xb, wb) = half(0.0, 1.0)
    global WN = vcat(xa, xb)
    global WT = vcat(wa .* (1 .- abs.(xa)), wb .* (1 .- abs.(xb)))
end

censored_pmf(n, d::UnivariateDistribution) = max(sum(WT .* pdf.(d, n .+ WN)), 1e-300)
nll(d, data) = -sum(log(censored_pmf(n, d)) for n in data)

make = Dict(
    :gamma     => p -> Gamma(p[1], p[2]),
    :lognormal => p -> LogNormal(log(p[1]), p[2]),
    :weibull   => p -> Weibull(p[1], p[2]))

moments = Dict(
    :gamma     => p -> (p[1]*p[2], sqrt(p[1])*p[2]),
    :lognormal => (p -> begin m = p[1]*exp(p[2]^2/2); (m, m*sqrt(exp(p[2]^2)-1)) end),
    :weibull   => (p -> begin d = Weibull(p[1], p[2]); (mean(d), std(d)) end))

function startp(fam, x)
    m = mean(x); v = var(x); cv2 = v/m^2
    fam == :gamma     ? [m^2/v, v/m] :
    fam == :lognormal ? [m/sqrt(1+cv2), sqrt(log(1+cv2))] : [1.2, m/0.9]
end

function fit_family(fam, x)
    f = lp -> nll(make[fam](exp.(lp)), x)
    o = optimize(f, log.(startp(fam, x)), NelderMead(),
                 Optim.Options(g_tol = 1e-10, iterations = 5000))
    p = exp.(Optim.minimizer(o)); (m, s) = moments[fam](p)
    (par = p, mean = m, sd = s, cv = s/m, aic = 2*length(p) + 2*Optim.minimum(o))
end

function boot_mean(fam, x; B = 1000)
    ms = Float64[]
    for _ in 1:B
        xb = rand(x, length(x))
        try push!(ms, fit_family(fam, xb).mean) catch end
    end
    quantile(ms, [0.025, 0.975])
end

pairs = CSV.read("../data/marburg_transmission_pairs.csv", DataFrame)
si = Float64.(Dates.value.(Date.(pairs.infectee_onset) .- Date.(pairs.infector_onset)))

sets = ["high only"    => pairs.confidence .== "high",
        "high+medium"  => in.(pairs.confidence, Ref(["high","medium"])),
        "all incl low" => trues(nrow(pairs))]

out = DataFrame()
for (nm, idx) in sets
    x = si[idx]
    fits = Dict(fam => fit_family(fam, x) for fam in keys(make))
    g = fits[:gamma]; ci = boot_mean(:gamma, x)
    best = argmin(Dict(fam => fits[fam].aic for fam in keys(fits)))
    push!(out, (set = nm, n_pairs = length(x),
        mean_censored = round(g.mean, digits=2), sd_censored = round(g.sd, digits=2),
        cv_censored = round(g.cv, digits=3),
        mean_lo = round(ci[1], digits=2), mean_hi = round(ci[2], digits=2),
        gamma_shape = round(g.par[1], digits=3), gamma_scale = round(g.par[2], digits=3),
        sd_naive = round(std(x), digits=2), cv_naive = round(std(x)/mean(x), digits=3),
        best_family = String(best),
        aic_gamma = round(fits[:gamma].aic, digits=2),
        aic_lognormal = round(fits[:lognormal].aic, digits=2),
        aic_weibull = round(fits[:weibull].aic, digits=2)); cols = :union)
end
CSV.write("../outputs/serial_interval_summary.csv", out)
show(out, allrows = true, allcols = true); println()
