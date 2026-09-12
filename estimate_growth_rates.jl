# Julia counterpart of estimate_growth_rates.R.
# Estimate each Marburg outbreak's intrinsic growth rate with a Poisson
# log-linear model glm(cases ~ t), t = onset/report date - start of outbreak,
# fit over the ascending phase of the zero-filled daily incidence series.
# See estimate_growth_rates.R for the full method description.
#
# Requires: CSV, DataFrames, GLM, Distributions (pin via Project.toml/Manifest.toml).
# Poisson GLM fitting (IRLS) is deterministic; seed kept for repo convention.

using CSV, DataFrames, GLM, Dates, Distributions, Statistics
using Random
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
]

parse_dmy(x)  = tryparse(Date, strip(String(x)), dateformat"d/m/y")   # 2/4/2005
parse_dbY(x)  = tryparse(Date, strip(String(x)), dateformat"d-u-y")   # 22-Nov-98

# Complete zero-filled daily incidence series as (dates, counts).
function daily_series(file, kind)
    df = CSV.read(file, DataFrame; normalizenames = false, stringtype = String)
    if kind == :agg
        d = parse_dmy.(string.(df[!, "Reported date"]))
        c = [tryparse(Float64, strip(string(v))) for v in df[!, "new cases"]]
        keep = .!isnothing.(d) .& .!isnothing.(c)
        dd = Date.(d[keep]); cc = Int.(round.(Float64.(c[keep])))
        g = combine(groupby(DataFrame(date = dd, cases = cc), :date), :cases => sum => :cases)
        sort!(g, :date)
        return g.date, g.cases
    end
    raw = kind == :drc ? parse_dbY.(string.(df.DT_ONSET)) : parse_dmy.(string.(df.ONSET_DATE))
    d = Date.(filter(!isnothing, raw))
    isempty(d) && return Date[], Int[]
    full = collect(minimum(d):Day(1):maximum(d))
    counts = [count(==(day), d) for day in full]
    return full, counts
end

rows = NamedTuple[]
for (label, file, kind) in outbreaks
    dates, cases = daily_series(file, kind)
    total = sum(cases; init = 0)
    note_kind = kind == :agg ?
        "aggregate surveillance (irregular reporting; new cases per report)" : ""

    if isempty(cases)
        push!(rows, (; outbreak = label, total_cases = total, n_days_full = 0,
            n_days_growth_phase = missing, cases_growth_phase = missing,
            growth_rate_per_day = missing, std_error = missing, ci_low = missing,
            ci_high = missing, doubling_time_days = missing, note = "no parseable dates"))
        continue
    end

    peak_val = maximum(cases)
    peak_pos = findlast(==(peak_val), cases)      # LAST day at maximum incidence
    wcases = cases[1:peak_pos]
    wdates = dates[1:peak_pos]
    ndays = length(wcases); cwin = sum(wcases)

    if peak_val < 2 || ndays < 3
        reason = peak_val < 2 ?
            "point-source / no growth phase: peak daily incidence <= 1" : "growth phase < 3 days"
        note = join(filter(!isempty, [note_kind, reason]), "; ")
        push!(rows, (; outbreak = label, total_cases = total, n_days_full = length(cases),
            n_days_growth_phase = ndays, cases_growth_phase = cwin,
            growth_rate_per_day = missing, std_error = missing, ci_low = missing,
            ci_high = missing, doubling_time_days = missing, note = note))
        continue
    end

    t = Float64.(Dates.value.(wdates .- minimum(wdates)))
    m = glm(@formula(cases ~ t), DataFrame(cases = wcases, t = t), Poisson(), LogLink())
    r  = coef(m)[2]
    se = stderror(m)[2]
    push!(rows, (; outbreak = label, total_cases = total, n_days_full = length(cases),
        n_days_growth_phase = ndays, cases_growth_phase = cwin,
        growth_rate_per_day = round(r, digits = 4), std_error = round(se, digits = 4),
        ci_low = round(r - Z * se, digits = 4), ci_high = round(r + Z * se, digits = 4),
        doubling_time_days = r > 0 ? round(log(2) / r, digits = 1) : missing, note = note_kind))
end

res = DataFrame(rows)
CSV.write("outbreak_growth_rates.csv", res; missingstring = "")
show(res, allrows = true, allcols = true); println()
