# Julia counterpart of estimate_serial_interval.R.
# Empirical Marburg serial interval from reconstructed infector-infectee pairs
# in marburg_transmission_pairs.csv; gamma fit by method of moments.
# See estimate_serial_interval.R for pair provenance and caveats
# (serial interval != generation interval; the high-tier variance is
# underestimated because most pairs share the Tanzania index case).

using CSV, DataFrames, Dates, Statistics

pairs = CSV.read("marburg_transmission_pairs.csv", DataFrame)
pairs.serial_interval_days = Dates.value.(Date.(pairs.infectee_onset) .- Date.(pairs.infector_onset))

function summ(x)
    x = Float64.(x); m = mean(x); v = var(x)
    (n_pairs = length(x), mean = round(m, digits=2), sd = round(std(x), digits=2),
     cv = round(std(x)/m, digits=3), median = round(median(x), digits=1),
     gamma_shape = round(m^2/v, digits=2), gamma_scale = round(v/m, digits=2),
     gamma_mean = round(m, digits=2))
end

sets = ["high only"    => pairs.confidence .== "high",
        "high+medium"  => in.(pairs.confidence, Ref(["high","medium"])),
        "all incl low" => trues(nrow(pairs))]

out = DataFrame()
for (nm, idx) in sets
    push!(out, merge((set = nm,), summ(pairs.serial_interval_days[idx])); cols = :union)
end
CSV.write("serial_interval_summary.csv", out)
show(out, allrows = true, allcols = true); println()
