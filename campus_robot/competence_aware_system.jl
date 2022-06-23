import Combinatorics
import Base: GLOBAL_RNG, isslotfilled, rand
function rand(r, s::Set)
    isempty(s) && throw(ArgumentError("set must be non-empty"))
    n = length(s.dict.slots)
    while true
        i = rand(r, 1:n)
        isslotfilled(s.dict, i) && return s.dict.keys[i]
    end
end
rand(s::Set) = rand(Base.GLOBAL_RNG, s)

using Plots
# using GLM
using DecisionTree
using DataFrames
using CSV
using JLD
using StatsBase

include("domain_model.jl")
include("../LAOStarSolver.jl")
include("../utils.jl")
include("../ValueIterationSolver.jl")

struct CASstate
    state::DomainState
        σ::Char
end

struct CASaction
    action::DomainAction
         l::Int
end

##

struct AutonomyModel
    L::Vector{Int}
    κ::Dict{Int, Dict{Int, Vector{Int}}}
    μ
end

function generate_autonomy_profile(𝒟::DomainSSP,
                                   L::Vector{Int})
    κ = Dict{Int, Dict{Int, Vector{Int}}}()
    for (s, state) in enumerate(𝒟.S)
        κ[s] = Dict{Int, Vector{Int}}()
        for (a, action) in enumerate(𝒟.A)
            if state.o != '∅'
                κ[s][a] = 1
            else
                κ[s][a] = 2
            end
        end
    end
    return κ
end

function update_potential(C, ℒ, s, a, L)
    state = CASstate(C.𝒮.D.S[s], '∅')
    s2 = C.SIndex[state]
    X = [lookahead(ℒ, s2, ((a - 1) * 4 + l + 1) ) for l ∈ L]
    P = softmax(-1.0 .* X)
    for l=1:size(L)[1]
        C.potential[s][a][L[l]+1] += P[l]
    end
    clamp!(C.potential[s][a], 0.0, 1.0)
end

function update_autonomy_profile!(C, ℒ)
    κ = C.𝒮.A.κ
    for (s, state) in enumerate(C.𝒮.D.S)
        for (a, action) in enumerate(C.𝒮.D.A)
            if κ[s][a] == competence(state, action)
                continue
            end

            if κ[s][a] == 0
                L = [0,1]
            elseif κ[s][a] == 3
                L = [2,3]
            else
                L = [κ[s][a]-1, κ[s][a], κ[s][a]+1]
            end
            update_potential(C, ℒ, s, a, L)

            distr = softmax([C.potential[s][a][l+1] for l in L])
            i = sample(aweights(distr))
            if L[i] == 3
                if C.𝒮.F.λ[s][a][2]['∅'] < 0.85
                    C.potential[s][a][L[i] + 1] = 0.0
                    continue
                end
            elseif L[i] == 0
                if C.𝒮.F.λ[s][a][1]['⊕'] > 0.25
                    C.potential[s][a][L[i] + 1] = 0.0
                    continue
                end
            elseif L[i] == κ[s][a]
                C.potential[s][a][L[i] + 1] = 0.0
                continue
            end

            if L[i] == competence(state, action)
                println("Updated to competence: ($s, $a) | $(κ[s][a]) | $(L[i])")
            end

            C.𝒮.A.κ[s][a] = L[i]
            C.potential[s][a][L[i] + 1] = 0.0
            if L[2] == 1 && L[i] == 2
                C.flags[s][a] = true
            end
        end
    end
end

function competence(state::DomainState,
                   action::DomainAction)
    if state.o == '∅'  || state.o == 'O' || state.o ==
        return 3
    elseif state.o == 'O'

    end
end

function save_autonomy_profile(κ)
    save(joinpath(abspath(@__DIR__),"params.jld"), "κ", κ)
end

function load_autonomy_profile()
    return load(joinpath(abspath(@__DIR__), "params.jld"), "κ")
end


function autonomy_cost(state::CASstate)
    if state.σ == '⊕' || state.σ == '∅'
        return 0.0
    elseif state.σ == '⊖'
        return 1.0
    elseif state.σ == '⊘'
        return 3.5
    end
end
##


##
struct FeedbackModel
    Σ::Vector{Char}
    λ::Dict{Int, Dict{Int, Dict{Int, Dict{Char, Float64}}}}
    ρ::Function
    D::Dict{String, Dict{String, DataFrame}}
    ϵ::Float64
end

function set_consistency(F::FeedbackModel, ϵ=0.95)
    F.ϵ=ϵ
end

function get_state_features(state::DomainState)
    #TODO
end

function generate_feedback_profile(𝒟::DomainSSP,
                                   Σ::Vector{Char},
                                   L::Vector{Int},
                                   D::Dict{String, Dict{String, DataFrame}})
    S, A = 𝒟.S, 𝒟.A
    λ = Dict(s => Dict(a => Dict(l => Dict(σ => 0.5 for σ ∈ Σ)
                                                   for l ∈ 1:2)
                                                   for a=1:length(A))
                                                   for s=1:length(S))
    for (a, action) in enumerate(A)
        X, Y = split_data(D[string(action.value)])
        M = build_forest(Y, X, 2, 10, 0.5, 8)

        for (s, state) in enumerate(S)
            f = get_state_features(state)
            for l ∈ [1,2]
                pred = apply_forest_proba(M, hcat(f, l), [0,1])
                for σ ∈ Σ
                    if σ == '⊖' || σ == '⊘'
                        λ[s][a][l][σ] = pred[1]
                    else
                        λ[s][a][l][σ] = pred[2]
                    end
                end
            end
        end
    end
    return λ
end

function update_feedback_profile!(C)
    λ, 𝒟, Σ, L, D = C.𝒮.F.λ, C.𝒮.D, C.𝒮.F.Σ, C.𝒮.A.L, C.𝒮.F.D
    S, A = 𝒟.S, 𝒟.A
    for (a, action) in enumerate(A)
        X, Y = split_data(D[string(action.value)])
        M = build_forest(Y, X, 2, 10, 0.5, 8)

        for (s, state) in enumerate(S)
            f = get_state_features(state)
            for l ∈ [1,2]
                pred = apply_forest_proba(M, hcat(f, l), [0,1])
                for σ ∈ Σ
                    if σ == '⊖' || σ == '⊘'
                        λ[s][a][l][σ] = pred[1]
                    else
                        λ[s][a][l][σ] = pred[2]
                    end
                end
            end
        end
    end
end

function save_feedback_profile(λ)
    save(joinpath(abspath(@__DIR__),"params.jld"), "λ", λ)
end

function load_feedback_profile()
    return load(joinpath(abspath(@__DIR__), "params.jld", "λ"))
end

function save_data(D)
    for k in keys(D)
        record_data(D[k], joinpath(abspath(@__DIR__), "data", "$k.csv"), false)
    end
end

function human_cost(action::CASaction)
    return [5. 1.5 5. 0.][action.l + 1]
end
##

struct CAS
    D::DomainSSP
    A::AutonomyModel
    F::FeedbackModel
end

struct CASSP
    𝒮::CAS
    S::Vector{CASstate}
    A::Vector{CASaction}
    T
    C
   s₀::CASstate
    G::Set{CASstate}
    SIndex::Dict{CASstate, Int}
    AIndex::Dict{CASaction, Int}
    flags::Dict{Int, Dict{Int, Bool}}
    potential::Dict{Int, Dict{Int, Vector{Float64}}}
end
function CASSP(𝒮::CAS,
               S::Vector{CASstate},
               A::Vector{CASaction},
               T::Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}},
               C::Function,
              s₀::CASstate,
               G::Set{CASstate})
    SIndex, AIndex = generate_index_dicts(S, A)
    flags = Dict(s => Dict(a => false for a=1:a_length) for s=1:s_length)
    potential = Dict(s => Dict(a => [0. for i=1:4] for a=1:a_length) for s=1:s_length)
    return CASSP(𝒮, S, A, T, C, s₀, G, SIndex, AIndex)
end

function generate_index_dicts(S::Vector{CASstate}, A::Vector{CASaction})
    SIndex = Dict{CASstate, Integer}()
    for (s, state) ∈ enumerate(S)
        SIndex[state] = s
    end
    AIndex = Dict{CASaction, Integer}()
    for (a, action) ∈ enumerate(A)
        AIndex[action] = a
    end
    return SIndex, AIndex
end

function generate_states(D, F)
    states = Vector{CASstate}()
    G = Set{CASstate}()
    for state in D.S
        for σ in F.Σ
            new_state = CASstate(state, σ)
            push!(states, new_state)
            if state in D.G
                push!(G, new_state)
            end
        end
    end
    return states, CASstate(D.s₀, '∅'), G
end

function reset_problem!(D, C)
    C.s₀ = CASstate(D.s₀, '∅')
    C.G = Set{CASstate}()
    for state in D.G
        for σ in C.𝒮.F.Σ
            push!(C.G, CASstate(state, σ))
        end
    end
end

function terminal(C::CASSP, state::CASstate)
    return state in C.G
end

function generate_actions(D, A)
    actions = Vector{CASaction}()
    for action in D.A
        for l in A.L
            new_action = CASaction(action, l)
            push!(actions, new_action)
        end
    end
    return actions
end

function allowed(C, s::Int, a::Int)
    return C.A[a].l <= C.𝒮.A.κ[ceil(s/4)][ceil(a/4)]
end

function generate_transitions(𝒟, 𝒜, ℱ,
                              S::Vector{CASstate},
                              A::Vector{CASaction},
                              G::Set{CASstate})

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
            if t == [(base_s, 1.0)]  || action.l > κ[base_s][base_a]
                T[s][a] = [(s, 1.0)]
                continue
            end

            p_approve = ℱ.λ[base_s][base_a][action.l]['⊕']
            p_disapprove = ℱ.λ[base_s][base_a][action.l]['⊖']
            p_override = ℱ.λ[base_s][base_a][action.l]['⊘']
            p_null = ℱ.λ[base_s][base_a][action.l]['∅']

            if action.l == 0
                T[s][a] = [((t[argmax([x[2] for x in t])][1]-1) * 4 + 4, 1.0)]
                # T[s][a] = transfer_control(𝒟, S, A, state, action)
                # for (sp, p) in t
                #     push!(T[s][a], (sp * 4 + 3, p))
                # end
            elseif action.l == 1
                push!(T[s][a], (base_s * 4 + 1, p_disapprove))
                for (sp, p) in t
                    push!(T[s][a], (sp * 4, p * p_approve))
                end
            elseif action.l == 2
                # push!(T[s][a], (base_s * 4 + 2, p_override))
                append!(T[s][a], ( (x-1, y * p_override) for (x,y) in T[s][C.AIndex[CASaction(action.action, 0)]]))
                for (sp, p) in t
                    push!(T[s][a], (sp * 4 + 4, p * p_null))
                end
            else
                for (sp, p) in t
                    push!(T[s][a], (sp * 4 + 4, p))
                end
            end
        end
    end

    return T
end

# function generate_cas_probability(state::DomainState,
#                                  action::DomainAction,
#                                       σ::Char)
#     if l == 0
#         return 1.0
#     elseif l == 1 && (σ == '⊘' || σ == '∅')
#         return 0.0
#     elseif l == 2 && (σ == '⊕' || σ == '⊖')
#         return 0.0
#     else
#
#     end
#     λ = ℱ.λ
#     p = λ(state, action, σ)
#
# end

function check_transition_validity(C)
    S, A, T = C.S, C.A, C.T
    for (s, state) in enumerate(S)
        for (a, action) in enumerate(A)
            mass = 0.0
            for (s′, p) in T[s][a]
                mass += p
                if p < 0.0
                    println("Transition error at state index $s and action index $a")
                    println("with a negative probability of $p.")
                    println("State: $(S[i])")
                    println("Action: $(A[j])")
                    @assert false
                end
            end
            if round(mass; digits=4) != 1.0
                println("Transition error at state $state and action $action.")
                println("State index: $s      Action index: $a")
                println("Total probability mass of $mass.")
                println("Transition vector is the following: $(T[s][a])")
                println("Succ state vector: $([S[s] for (s,p) in T[s][a]])")
                @assert false
            end
        end
    end
end

function block_transition!(C::CASSP,
                       state::CASstate,
                      action::CASaction)
    state′ = CASstate(state.state, '⊕')
    s, a = C.SIndex[state′], C.AIndex[action]
    # T[s][a] = [(s, 1.0)]
    C.T[s+1][a] = [(s+1, 1.0)]
    C.T[s+2][a] = [(s+2, 1.0)]
    C.T[s+3][a] = [(s+3, 1.0)]
end

function generate_costs(C::CASSP, s::Int, a::Int,)
    D, A, F = C.𝒮.D, C.𝒮.A, C.𝒮.F
    state, action = C.S[s], C.A[a]
    cost = D.C(D, D.SIndex[state.state], D.AIndex[action.action])
    cost += A.μ(state, action)
    cost += F.ρ(state, action)
    return cost
end

function generate_feedback(C::CASSP, state::CASstate, action::CASaction)
    if rand() <= 0.1
        if action.l == 1
            return ['⊕', '⊖'][rand(1:2)]
        elseif action.l == 2
            return ['∅', '⊘'][rand(1:2)]
        end
    end

    info = get_state_info(state)

    if info[:obstacle] == :door
        if info[:doortype] == :pull || info[:doorsize] == :large
            return (action.l == 1) ? '⊖' : '⊘'
        else
            if info[:doorsize] == :small || (info[:doorsize] == :medium && info[:region] in [:b1, :b2])
                return (action.l == 1) ? '⊕' : '∅'
            else
                return (action.l == 1) ? '⊖' : '⊘'
            end
        end
    elseif info[:obstacle] == :crosswalk
        if info[:traffic] == :empty || (info[:traffic] == :light && info[:visibility] == :high && info[:pedestrian] == false)
            return (action.l == 1) ? '⊕' : '∅'
        else
            return (action.l == 1) ? '⊖' : '⊘'
        end
    else

    end
end

function build_cas(𝒟::DomainSSP,
                   L::Vector{Int},
                   Σ::Vector{Char})
    κ = generate_autonomy_profile(𝒟, L)
    𝒜 = AutonomyModel(L, κ, autonomy_cost)
    λ = generate_feedback_profile(𝒟, Σ, L)
    ℱ = FeedbackModel(Σ, λ, human_cost)
    𝒮 = CAS(𝒟, 𝒜, ℱ)
    S, s₀, G = generate_states(𝒟, ℱ)
    A = generate_actions(𝒟, 𝒜)
    T = generate_transitions(𝒟, 𝒜, ℱ, S, A, G)

    𝒮 = CASSP(𝒮, S, A, T, generate_costs, s₀, G)
end

function solve_model(C::CASSP)
    ℒ = LAOStarSolver(100000, 1000., 1.0, .001, Dict{Integer, Integer}(),
                        zeros(length(C.S)), zeros(length(C.S)),
                        zeros(length(C.S)), zeros(length(C.A)))
    a, total_expanded = solve(ℒ, C, C.SIndex[C.s₀])
    println("LAO* expanded $total_expanded nodes.")
    println("Expected cost to goal: $(ℒ.V[M.SIndex[M.s₀]])")
    return ℒ
end

function main()
    M = build_model()
    C = build_cas(M, [0,1,2,3], ['⊕', '⊖', '⊘', '∅'])
    ℒ = solve_model(C)
end

main()
