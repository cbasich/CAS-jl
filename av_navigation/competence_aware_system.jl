import Combinatorics

using GLM
using DataFrames
using CSV
using HDF5, JLD

include("domain_model.jl")
include("../LAOStarSolver.jl")
include("../utils.jl")

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
            if typeof(state) == NodeState
                if state.p == true || state.o == true || state.v > 1
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

function autonomy_cost(state::CASstate)
    if state.σ == '⊕' || state.σ == '∅'
        return 0.0
    elseif state.σ == '⊖'
        return 1.0
    elseif state.sigma == '⊘'
        return 3.0
    end
end
##


##
struct FeedbackModel
    Σ::Vector{Char}
    λ::Dict{Int, Dict{Int, Dict{Int, Dict{Char, Float64}}}}
    ρ::Function
end

function get_state_features(state)
    if typeof(state) == NodeState
        return [state.p state.o state.v]
    else
        return [state.o state.l]
    end
end

function generate_feedback_profile(𝒟::DomainSSP,
                                   Σ::Vector{Char},
                                   L::Vector{Int})
    λ = Dict{Int, Dict{Int, Dict{Int, Dict{Char, Float64}}}}()
    for (s, state) in enumerate(𝒟.S)
        f = get_state_features(state)
        λ[s] = Dict{Int, Dict{Int, Dict{Char, Float64}}}()
        for (a, action) in enumerate(𝒟.A)
            if typeof(state) == NodeState
                X, Y = read_data(joinpath(abspath(@__DIR__), "data", "node_$(action.value).csv"))
                fm = @formula(y ~ x1 + x2 + x3 + x4)
            else
                if action.value ∉ ['↑', '⤉']
                    continue
                end
                X, Y = read_data(joinpath(abspath(@__DIR__), "data", "edge_$(action.value).csv"))
                fm = @formula(y ~ x1 + x2 + x3)
            end

            insufficient_data = false
            try
                logit = glm(fm, hcat(X, Y), Binomial(), ProbitLink())
            catch
                insufficient_data = true
            end

            λ[s][a] = Dict{Int, Dict{Char, Float64}}()
            for l in L
                λ[s][a][l] = Dict{Char, Float64}()
                for σ ∈ Σ
                    if insufficient_data
                        λ[s][a][l][σ] = 0.5
                        continue
                    end

                    q = DataFrame(hcat(f, l), :auto)
                    p = 0.5
                    try
                        p = predict(logit, q)[1]
                    catch
                        p = 0.5
                    end
                    if σ == '⊕' || σ == '∅'
                        λ[s][a][l][σ] = p
                    else
                        λ[s][a][l][σ] = 1.0 - p
                    end
                end
            end
        end
    end
    return λ
end

function save_feedback_profile(λ)
    save(joinpath(abspath(@__DIR__),"params.jld"), "λ", λ)
end

function human_cost(action::CASaction)
    return [3 2 1 0][action.l + 1]              #TODO: Fix this.
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
end
function CASSP(𝒮::CAS,
               S::Vector{CASstate},
               A::Vector{CASaction},
               T::Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}},
               C::Function,
              s₀::CASstate,
               G::Set{CASstate})
    SIndex, AIndex = generate_index_dicts(S, A)
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

function generate_transitions!(𝒟, 𝒜, ℱ, C,
                              S::Vector{CASstate},
                              A::Vector{CASaction},
                              G::Set{CASstate})

    T = C.T
    κ, λ = 𝒜.κ, ℱ.λ
    for (s, state) in enumerate(S)
        T[s] = Dict{Int, Vector{Tuple{Int, Float64}}}()
        for (a, action) in enumerate(A)
            if state in G
                state′ = CASstate(state.state, '∅')
                T[s][a] = [(C.SIndex[state′], 1.0)]
                continue
            end

            base_state = state.state
            base_action = action.action
            base_s = 𝒟.SIndex[base_state]
            base_a = 𝒟.AIndex[base_action]

            t = 𝒟.T[base_s][base_a]
            if t == [(base_s, 1.0)] || action.l ∉ κ[base_s][base_a]
                T[s][a] = [(s, 1.0)]
                continue
            end

            T[s][a] = Vector{Tuple{Int, Float64}}()
            if action.l == 0
                T[s][a] = [((t[argmax([x[2] for x in t])][1]-1) * 4 + 4 , 1.0)]
            elseif action.l == 1
                p_approve = λ[base_s][base_a][1]['⊕']
                p_disapprove = 1.0 - p_approve #λ[base_s][base_a][1]['⊖']
                push!(T[s][a], ((base_s-1) * 4 + 2, p_disapprove))
                for (sp, p) in t
                    push!(T[s][a], ((sp-1) * 4 + 1, p * p_approve))
                end
            elseif action.l == 2
                p_override = λ[base_s][base_a][2]['⊘']
                p_null = 1.0 - p_override #λ[base_s][base_a][2]['∅']
                append!(T[s][a], ( (x, y * p_override) for (x,y) in T[s][C.AIndex[CASaction(action.action, 0)]]))
                for (sp, p) in t
                    push!(T[s][a], ((sp-1) * 4 + 4, p * p_null))
                end
            else
                for (sp, p) in t
                    push!(T[s][a], ((sp-1) * 4 + 4, p))
                end
            end
        end
    end
end

function generate_costs(C::CASSP,
                        s::Int,
                        a::Int,)
    D, A, F = C.𝒮.D, C.𝒮.A, C.𝒮.F
    state, action = C.S[s], C.A[a]
    cost = D.C(D, D.SIndex[state.state], D.AIndex[action.action])
    cost += A.μ(state)
    cost += F.ρ(action)
    return cost
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
    T = Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}}()

    C = CASSP(𝒮, S, A, T, generate_costs, s₀, G)
    generate_transitions!(𝒟, 𝒜, ℱ, C, S, A, G)

    return C
end

function solve_model(C::CASSP)
    ℒ = LAOStarSolver(100000, 1000., 1.0, .001, Dict{Integer, Integer}(),
                        zeros(length(C.S)), zeros(length(C.S)),
                        zeros(length(C.S)), zeros(length(C.A)))
    a, total_expanded = solve(ℒ, C, C.SIndex[C.s₀])
    println("LAO* expanded $total_expanded nodes.")
    println("Expected cost to goal: $(ℒ.V[C.SIndex[C.s₀]])")
    return ℒ
end


function init_data()
    for action in ["←", "↑", "→", "⤉"]
        init_node_data(joinpath(abspath(@__DIR__), "data", "node_$action.csv"))
    end

    init_edge_data(joinpath(abspath(@__DIR__), "data", "edge_↑.csv"))
    init_edge_data(joinpath(abspath(@__DIR__), "data", "edge_⤉.csv"))
end

function main()
    init_data()

    M = build_model()
    C = build_cas(M, [0,1,2,3], ['⊕', '⊖', '⊘', '∅'])
    ℒ = solve_model(C)

    save_feedback_profile(C.𝒮.F.λ)
end

@show load(joinpath(abspath(@__DIR__), "params.jld"), "λ")

main()
