
using Plots
using Integrals
using DataFrames
using Base.Threads
using CSV


function ρ(x::Float64)
    return sin(x)^2 * exp(-x)
end

function kernel_1(x::Float64, k::Float64)
    return 1 / (k^2 + x^2)
end

function kernel_2(x::Float64, k::Float64)
    return 1 / (1 + k^2*x^2)
end

function kernel_3(x::Float64, k::Float64)
    return sin(x) / (k^2 + x^2)
end

function kernel_4(x::Float64, k::Float64)
    return k * exp(-x)
end

function kernel_5(x::Float64, k::Float64)
    return exp(-(k*x)^2)
end

function integral_1(x, k)
    return ρ.(x) .* kernel_1.(x, k)
end

function integral_2(x, k)
    return ρ.(x) .* kernel_2.(x, k)
end

function integral_3(x, k)
    return ρ.(x) .* kernel_3.(x, k)
end

function integral_4(x, k)
    return ρ.(x) .* kernel_4.(x, k)
end

function integral_5(x, k)
    return ρ.(x) .* kernel_5.(x, k)
end

function g(k, integral)
    prob = IntegralProblem(integral, 0, 1000, k)
    sol = solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
    return sol.u
end

function g_array(ks, integral)
    gs = zeros(Float64, length(ks))
    for (ind, k) in enumerate(ks)
        gs[ind] = g(k, integral)
    end
    return gs
end


function calculate_all_integrals(ks::Array, integrals)
    data = zeros(Float64, (length(ks), length(integrals)))
    for (ind, integral) in enumerate(integrals)
        data[:, ind] = g_array(ks, integral)
    end
    df = DataFrame(data, :auto)

    return df
end

filepath = "integrals_feature.csv"
ks = collect(0:1e-4:10)

integrals = [
    integral_1, integral_2, integral_3, integral_4, integral_5
]


df = calculate_all_integrals(ks, integrals)
CSV.write(filepath, df)

#for i in range(1, length(integrals), length(integrals))
#    i = Int(i)
#    plot!(ks, df[:, i], label=string(i), xlabel="k", ylabel="y")
#end
# plot!(x, integrand_1, label="i1")
#savefig("integrals.png")

