# load_ebola_data.jl
# Julia Script

#=
Description:
Author: zhangjingyan
Date: 05/08/2026
=#

using XLSX
using DataFrames
using Dates
using Tables

"""
    load_ebola_data(filename; country, indicator_keyword)

Load Ebola data and return:
    t       -> Vector{Float64}
    I_data  -> Vector{Float64}

Arguments:
    filename:
        path to ebola_data_db_format.xlsx

    country:
        e.g. "Guinea", "Liberia", "Sierra Leone"

    indicator_keyword:
        part of the indicator name to select.
        Default selects cumulative confirmed/probable/suspected CASES
        (not deaths). NOTE: this must be specific enough to match only
        one indicator, since the dataset also contains a "...Ebola
        deaths" indicator that shares the same "confirmed, probable
        and suspected" prefix.
"""
function load_ebola_data(
    filename;
    country="Guinea",
    indicator_keyword="Cumulative number of confirmed, probable and suspected Ebola cases"
)

    xf = XLSX.readxlsx(filename)

    sheetname = XLSX.sheetnames(xf)[1]

    data = XLSX.readtable(filename, sheetname)

    df = DataFrame(data)

    # select country
    df = filter(row -> row.Country == country, df)

    # select indicator
    df = filter(
        row -> occursin(indicator_keyword, row.Indicator),
        df
    )

    # safety check: indicator_keyword should resolve to exactly one
    # indicator string. If it matches more than one (e.g. both
    # "...cases" and "...deaths"), the later dedup step would silently
    # keep whichever one happens to sort last per date.
    matched_indicators = unique(df.Indicator)
    if length(matched_indicators) > 1
        error(
            "indicator_keyword \"$indicator_keyword\" matches multiple " *
            "indicators: $matched_indicators. Make indicator_keyword " *
            "more specific."
        )
    end

    # convert date
    df.Date = Date.(df.Date)

    sort!(df, :Date)

    # remove duplicated dates (e.g. multiple reports on the same date)
    df = combine(
        groupby(df, :Date),
        :value => last => :value
    )

    t = collect(0.0:length(df.value)-1)

    I_data = Float64.(df.value)

    return t, I_data
end