# load_covid_data.jl
# Julia Script

#=
Description:
Author: zhangjingyan
Date: 05/08/2026
=#

using XLSX
using DataFrames
using Dates

"""
    load_covid_data(filename; country, start_date=nothing)

Read ECDC COVID dataset.

Returns:
    t      : Vector{Float64}
    I_data : cumulative cases

Arguments:
    filename:
        path to ECDC excel file

    country:
        e.g. "Japan", "Sweden"

    start_date:
        date where t=0 begins.
        Example: Date(2020,1,30)

        If nothing, use first available date.
"""
function load_covid_data(
    filename;
    country="Japan",
    start_date=nothing
)

    df = DataFrame(
        XLSX.readtable(filename, 1)
    )

    # select country
    df = filter(
        row -> row.countriesAndTerritories == country,
        df
    )

    # convert date
    # NOTE: XLSX stores dateRep as a real date-formatted cell, so
    # XLSX.readtable already returns Date/DateTime objects, not strings.
    # Calling Date.(df.dateRep, dateformat"yyyy-mm-dd") errors in that case
    # (Date(::DateTime, ::DateFormat) has no method). This helper handles
    # both cases so it keeps working if a differently-formatted file
    # (with dateRep as plain text) is used later.
    to_date(x::AbstractString) = Date(x, dateformat="yyyy-mm-dd")
    to_date(x::Union{Date,DateTime}) = Date(x)
    df.dateRep = to_date.(df.dateRep)

    # sort oldest -> newest
    sort!(df, :dateRep)


    # choose starting date
    if start_date !== nothing
        df = filter(
            row -> row.dateRep >= start_date,
            df
        )
    end


    # cumulative infected
    I_data = cumsum(Float64.(df.cases))


    # time vector
    t = collect(0.0:length(I_data)-1)


    return t, I_data
end