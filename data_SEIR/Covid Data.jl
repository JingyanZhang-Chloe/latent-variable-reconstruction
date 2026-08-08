# Covid Data.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 05/08/2026
=#

include("../src/SEIR/SEIRModels.jl")
include("../src/SEIR/weak_form.jl")
include("load_covid_data.jl")
using .Value
using .Logic
using HomotopyContinuation

@var αT, σT, γT, S0, E0
const variables = [αT, σT, γT, S0, E0]


function main()
    t_Japan, I_Japan = load_covid_data(
        "data_SEIR/COVID-19-geographic-disbtribution-worldwide-2020-12-14.xlsx",
        country="Japan",
        start_date=Date(2020,1,30)
    )

    t_Sweden, I_Sweden = load_covid_data(
        "data_SEIR/COVID-19-geographic-disbtribution-worldwide-2020-12-14.xlsx",
        country="Sweden",
        start_date=Date(2020,2,1)
    )

    println(length(t_Japan))
    println(length(t_Sweden))

end

main()
