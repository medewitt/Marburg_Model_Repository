# Julia counterpart of estimate_growth_rates.R.
# Intrinsic growth rate of each Marburg outbreak from the ascending phase of the
# zero-filled daily incidence series, with a NEGATIVE BINOMIAL observation model
# (primary) and Poisson as a sensitivity fit.
#
#   negbin(@formula(cases ~ t), df, LogLink())   # Var = mu + mu^2/theta
#
# r is the slope on t; 95% Wald CI and doubling time ln(2)/r. A likelihood-ratio
# test of Poisson vs NB (boundary 0.5 chi^2_1) gives overdisp_p. Where the NB
# dispersion is not identified (fit fails or SE unstable) the NB columns fall
# back to Poisson and the note records it. See estimate_growth_rates.R for the
# full method. Requires CSV, DataFrames, GLM, Distributions (pin versions).
# Seed 1834 kept for repo convention (fitting is deterministic).

using CSV, DataFrames, GLM, Dates, Distributions, Statistics, Random
Random.seed!(1834)

const Z = quantile(Normal(), 0.975)

outbreaks = [
    ("Angola 2005",       "MarburgAngola2005Data.csv",   :agg),
    ("Belgrade 1967",     "MarburgBelgrade1967Line.csv", :line),
    ("DRC 1999",          "MarburgDRC1999LineData.csv",  :drc),
    ("Frankfurt 1967",    "MarburgFrankfurt1967Line.csv",:line),
    ("Guinea 2021",       "MarburgGuinea2021Line.csv",   :line),
    ("Kenya 1980",        "MarburgKenya1980Line.csv",    :line),
    ("Kenya 1987",        "MarburgKenya1987Line.csv",    :line),
    ("Marburg 1967",      "MarburgMarburg1967Line.csv",  :line),
    ("South Africa 1975", "MarburgSA1975Line.csv",       :line),
    ("Uganda (ND) 2008",  "MarburgUG_ND2008Line.csv",    :line),
    ("Uganda (US) 2008",  "MarburgUG_US2008Line.csv",    :line),
    ("Uganda 2007",       "MarburgUganda2007Line.csv",   :line),
    ("Uganda 2012",       "MarburgUganda2012Line.csv",   :line),
    ("Uganda 2014",       "MarburgUganda2014Line.csv",   :line),
    ("Uganda 2017",       "MarburgUganda2017Line.csv",   :line),
    ("Tanzania 2023",     "MarburgTanzania2023Line.csv", :line),
    ("Tanzania 2023 (excl. index)", "MarburgTanzania2023Line.csv", :line),
    ("Rwanda 2024",       "MarburgRwanda2024Line.csv",   :line),
]

const DROP_INDEX = Set(["Tanzania 2023 (excl. index)"])
const NOTES = Dict(
    "Tanzania 2023" => "all cases; r inflated by 9-day gap between index case (onset 27 Feb) and cluster",
    "Tanzania 2023 (excl. index)" => "index case (onset 27 Feb) dropped; post-introduction human-to-human phase",
    "Rwanda 2024" => "66 lab-confirmed cases by onset, digitized from NEJM Fig 1 (Nsanzimana 2025); 2 probable Aug cases excluded",
)

parse_dmy(x) = tryparse(Date, strip(String(x)), dateformat"d/m/y")
parse_dbY(x) = tryparse(Date, strip(String(x)), dateformat"d-u-y")

function daily_series(file, kind; drop_index = false)
    df = CSV.read(joinpath("..", "data", file), DataFrame;
                  normalizenames = false, stringtype = String)
    if kind == :agg
        d = parse_dmy.(string.(df[!, "Reported date"]))
        c = [tryparse(Float64, strip(string(v))) for v in df[!, "new cases"]]
        keep = .!isnothing.(d) .& .!isnothing.(c)
        g = combine(groupby(DataFrame(date = Date.(d[keep]), cases = Int.(round.(Float64.(c[keep])))),
                            :date), :cases => sum => :cases)
        sort!(g, :date); return g.date, g.cases
    end
    raw = kind == :drc ? parse_dbY.(string.(df.DT_ONSET)) : parse_dmy.(string.(df.ONSET_DATE))
    d = Date.(filter(!isnothing, raw))
    drop_index && !isempty(d) && (d = filter(!=(minimum(d)), d))
    isempty(d) && return Date[], Int[]
    full = collect(minimum(d):Day(1):maximum(d))
    return full, [count(==(day), d) for day in full]
end

blank(label, total, ndf; ngp = missing, cgp = missing, note = "") = (;
    outbreak = label, total_cases = total, n_days_full = ndf,
    n_days_growth_phase = ngp, cases_growth_phase = cgp,
    r_nb = missing, se_nb = missing, ci_low_nb = missing, ci_high_nb = missing,
    doubling_nb = missing, nb_alpha = missing, overdisp_p = missing,
    r_poisson = missing, se_poisson = missing, ci_low_poisson = missing,
    ci_high_poisson = missing, note = note)

# extract theta (NB dispersion) from a GLM.jl negbin fit across versions
nb_theta(m) = try; m.model.rr.d.r; catch; try; m.rr.d.r; catch; missing; end; end

rows = NamedTuple[]
for (label, file, kind) in outbreaks
    dates, cases = daily_series(file, kind; drop_index = label in DROP_INDEX)
    total = sum(cases; init = 0)
    note_kind = join(filter(!isempty, [
        kind == :agg ? "aggregate surveillance (irregular reporting; new cases per report)" : "",
        get(NOTES, label, "")]), "; ")
    if isempty(cases)
        push!(rows, blank(label, total, 0; note = "no parseable dates")); continue
    end
    peak_val = maximum(cases); peak_pos = findlast(==(peak_val), cases)
    wcases = cases[1:peak_pos]; wdates = dates[1:peak_pos]
    ndays = length(wcases); cwin = sum(wcases)
    if peak_val < 2 || ndays < 3
        reason = peak_val < 2 ? "point-source / no growth phase: peak daily incidence <= 1" :
                 "growth phase < 3 days"
        push!(rows, blank(label, total, length(cases); ngp = ndays, cgp = cwin,
              note = join(filter(!isempty, [note_kind, reason]), "; "))); continue
    end

    t = Float64.(Dates.value.(wdates .- minimum(wdates)))
    df = DataFrame(cases = wcases, t = t)
    pois = glm(@formula(cases ~ t), df, Poisson(), LogLink())
    rp = coef(pois)[2]; sep = stderror(pois)[2]
    ci_p = (round(rp - Z*sep, digits=4), round(rp + Z*sep, digits=4))

    nb = try; negbin(@formula(cases ~ t), df, LogLink()); catch; nothing; end
    theta = nb === nothing ? missing : nb_theta(nb)
    senb = nb === nothing ? Inf : stderror(nb)[2]
    if nb !== nothing && isfinite(senb) && senb <= 10 && !ismissing(theta)
        rnb = coef(nb)[2]
        lr = 2 * (loglikelihood(nb) - loglikelihood(pois))
        push!(rows, (; outbreak = label, total_cases = total, n_days_full = length(cases),
            n_days_growth_phase = ndays, cases_growth_phase = cwin,
            r_nb = round(rnb, digits=4), se_nb = round(senb, digits=4),
            ci_low_nb = round(rnb - Z*senb, digits=4), ci_high_nb = round(rnb + Z*senb, digits=4),
            doubling_nb = rnb > 0 ? round(log(2)/rnb, digits=1) : missing,
            nb_alpha = round(1/theta, digits=4),
            overdisp_p = round(0.5 * ccdf(Chisq(1), max(lr, 0.0)), digits=4),
            r_poisson = round(rp, digits=4), se_poisson = round(sep, digits=4),
            ci_low_poisson = ci_p[1], ci_high_poisson = ci_p[2], note = note_kind))
    else
        push!(rows, (; outbreak = label, total_cases = total, n_days_full = length(cases),
            n_days_growth_phase = ndays, cases_growth_phase = cwin,
            r_nb = round(rp, digits=4), se_nb = round(sep, digits=4),
            ci_low_nb = ci_p[1], ci_high_nb = ci_p[2],
            doubling_nb = rp > 0 ? round(log(2)/rp, digits=1) : missing,
            nb_alpha = missing, overdisp_p = missing,
            r_poisson = round(rp, digits=4), se_poisson = round(sep, digits=4),
            ci_low_poisson = ci_p[1], ci_high_poisson = ci_p[2],
            note = join(filter(!isempty, [note_kind, "NB alpha not identified; NB columns = Poisson fit"]), "; ")))
    end
end

res = DataFrame(rows)
CSV.write("../outputs/outbreak_growth_rates.csv", res; missingstring = "")
show(res, allrows = true, allcols = true); println()
