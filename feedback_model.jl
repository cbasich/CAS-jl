include("domain_model.jl")

struct FeedbackModel
    Σ::Vector{Char}
    λ::Dict{Int, Dict{Int, Dict{Char, Float64}}}
    ρ::Function
end
