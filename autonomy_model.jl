include("domain_model.jl")

struct AutonomyModel
    L::Vector{Int}
    κ::Dict{Int, Dict{Int, Vector{Int}}}
    μ::function
end
