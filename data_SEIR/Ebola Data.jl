# Ebola Data.jl
# Julia Script

#=
Description: 
Author: zhangjingyan
Date: 05/08/2026
=#

include("../src/SEIR/SEIRModels.jl")
include("../src/SEIR/weak_form.jl")
include("load_ebola_data.jl")
using .Value
using .Logic
using HomotopyContinuation

@var αT, σT, γT, S0, E0
const variables = [αT, σT, γT, S0, E0]


function main()
    t, I_data = load_ebola_data(
        "data_SEIR/ebola_data_db_format.xlsx",
        country="Guinea"
    )

    HC_LS_weak(t, I_data, variables, "CSpline_GK", :chebyshev_U, compare_LS=true, profiling=true, perturb=0.50)
end

main()
