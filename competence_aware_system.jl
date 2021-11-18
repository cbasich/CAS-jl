import Combinatorics

include("domain_model.jl")
include("LAOStarSolver.jl")

##

struct AutonomyModel
    L::Vector{Int}
    κ::Dict{Int, Dict{Int, Vector{Int}}}
    μ::function
end

function generate_autonomy_profile(𝒟::DomainSSP,
                                   L::Vector{int})
    κ = Dict{Int, Dict{Int, Vector{Int}}}()
    for (s, state) in enumerate(𝒟.S)
        κ[s] = Dict{Int, Vector{Int}}()
        for (a, action) in enumerate(𝒟.A)
            if typeof(state) == NodeState
                if state.p == true || state.v > 1
                    κ[s][a] = [0,1]
                else
                    κ[s][a] = L
                end
            else
                if state.o == true
                    κ[s][a] = [0,1]
                else
                    κ[s][a] = L
                end
            end
        end
    end
    return κ
end

function autonomy_cost(state::DomainState,
                      action::CASaction,
                           l::Int)
    return 1.0                          #TODO: Replace this wih the correct cost.
end
##


##
struct FeedbackModel
    Σ::Vector{Char}
    λ::Dict{Int, Dict{Int, Dict{Char, Float64}}}
    ρ::Function
end

function generate_feedback_profile(𝒟::DomainSSP,
                                   Σ::Vector{Char})
    λ = Dict{Int, Dict{Int, Dict{Char, Float64}}}()
    for (s, state) in enumerate(𝒟.S)
        λ[s] = Dict{Int, Dict{Char, Float64}}()
        for (a, action) in enumerate(𝒟.A)
            λ[s][a] = Dict{Char, Float64}()
            for σ ∈ Σ
                λ[s][a][σ] = get_feedback_probability(state, action, σ)
            end
        end
    end
    return λ
end

function get_feedback_probability(state::DomainState,
                                 action::DomainAction,
                                      σ::Char)
    F = get_feature_vector(state)

end

function human_cost(state::DomainState,
                   action::CASaction,
                        l::Int)
    return 1.0                            #TODO: Replace this with correct cost.
end
##

struct CAS
    D::DomainSSP
    A::AutonomyModel
    F::FeedbackModel
end

struct CASstate
    state::DomainState
        σ::Char
end

struct CASaction
    action::DomainAction
         l::Int
end

struct CASSP
    S::Vector{CASstate}
    A::Vector{CASaction}
    T
    C
   s₀::CASstate
    G::Vector{CASState}
end

function generate_states(𝒮::CAS)
    states = Vector{CASstate}
    for state in 𝒮.D.S
        for σ in 𝒮.F.Σ
            new_state = (state, σ)
            push!(states, new_state)
        end
    end
    return states
end

function generate_actions(𝒮::CAS)
    actions = Vector{CASaction}
    for action in 𝒮.D.A
        for l in 𝒮.A.L
            new_action = (action, l)
            push!(actions, new_action)
        end
    end
    return actions
end

function generate_transitions(𝒮::CAS,
                              S::Vector{CASstate},
                              A::Vector{CASaction},
                              G::Vector{CASstate})
    𝒟, 𝒜, ℱ = 𝒮.𝒟, 𝒮.𝒜, 𝒮.ℱ

    T = Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}}()
    for (s, state) in enumerate(S)
        T[s] = Dict{Int, Vector{Tuple{Int, Float64}}}()
        for (a, action) in enumerate(A)
            if state in G
                T[s][a] = [(s, 1.0)]
            end
            T[s][a] = Vector{Tuple{Int, Float64}}()
            base_state = state.state
            base_action = action.action
            base_s = 𝒟.SIndex[base_state]
            base_a = 𝒟.AIndex[base_action]

            t = 𝒟.T[base_s][base_a]
            for (sp, p) in t
                state′ = 𝒟.S[sp]
                for σ ∈ ℱ.Σ
                    new_p = p * ℱ.λ(state, action, σ, state′)
                    push!(T[s][a], (state′, new_p))
                end
            end
        end
    end

    return T
end

function generate_cas_probability(state::DomainState,
                                 action::DomainAction,
                                      σ::Char)
    if l == 0
        return 1.0
    elseif l == 1 && (σ == '⊘' || σ == '∅')
        return 0.0
    elseif l == 2 && (σ == '⊕' || σ == '⊖')
        return 0.0  
    else

    end
    λ = ℱ.λ
    p = λ(state, action, σ)

end

function generate_costs(𝒮::CAS
                        s::Int, state::CASstate,
                        a::Int, action::CASaction)
    D, A, F = 𝒮.D, 𝒮.A, 𝒮.F
    cost = D.C(D, D.SIndex[state.state], D.AIndex[action.action])
    cost += A.μ(state, action)
    cost += F.ρ(state, action)
    return cost
end

function build_cas(𝒟::DomainSSP,
                   L::Vector{Int},
                   Σ::Vector{Char})
    κ = generate_autonomy_profile(𝒟, L)
    λ = generate_feedback_profile(𝒟, Σ)
end
