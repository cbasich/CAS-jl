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
using JLD2
using StatsBase

include("domain_model.jl")
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
    κ::Dict{Int, Dict{Int, Int}}
    μ
end

function generate_autonomy_profile(𝒟::DomainSSP)
    κ = Dict{Int, Dict{Int, Int}}()
    for (s, state) in enumerate(𝒟.S)
        κ[s] = Dict{Int, Int}()
        for (a, action) in enumerate(𝒟.A)
            # κ[s][a] = 2
            if typeof(state) == EdgeState && action.value == '↑'
                κ[s][a] = 3
            else
                κ[s][a] = 2
            end
            # if typeof(state) == NodeState
            #     if action.value == '⤉' || action.value == '↓'
            #         κ[s][a] = 3
            #     elseif state.p == true || state.o == true || state.v > 1
            #         κ[s][a] = 1
            #     else
            #         κ[s][a] = 3
            #     end
            # else
            #     if state.o == true
            #         κ[s][a] = 1
            #     else
            #         κ[s][a] = 3
            #     end
            # end
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
        # solve(ℒ, C, s)
        for (a, action) in enumerate(C.𝒮.D.A)
            # if κ[s][a] == 3 || κ[s][a] == 0
            #     continue
            # end
            if κ[s][a] == competence(state, action)
                continue
            end
            if typeof(state) == EdgeState && action.value ∉ ['↑', '⤉']
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

            # for i in sortperm(-distr)
            #     t += distr[i]
            #
            #     if rand() <= C.potential[s][a][L[i]+1]
            #         if L[i] == 3
            #             if C.𝒮.F.λ[s][a][2]['∅'] < 0.65
            #                 C.potential[s][a][L[i] + 1] = 0.0
            #                 break
            #             end
            #         elseif L[i] == 0
            #             if C.𝒮.F.λ[s][a][1]['⊕'] > 0.35
            #                 C.potential[s][a][L[i] + 1] = 0.0
            #                 break
            #             end
            #         elseif L[i] == κ[s][a]
            #             C.potential[s][a][L[i] + 1] = 0.0
            #             break
            #         end
            #         if L[i] == competence(state, action)
            #             println("Updated to competence: ($s, $a) | $(κ[s][a]) | $(L[i])")
            #         end
            #         C.𝒮.A.κ[s][a] = L[i]
            #         C.potential[s][a][L[i] + 1] = 0.0
            #         if L[2] == 1 && L[i] == 2
            #             C.flags[s][a] = true
            #         end
            #         break
            #     end
            # end
        end
    end
end

function competence(state::DomainState,
                   action::DomainAction)
    if typeof(state) == EdgeState
        if state.o && state.l == 1
            return 0
        else
            return 3
        end
    else
        if action.value == '⤉'
            return 3
        elseif action.value == '→'
            if state.o && state.p && state.v > 1
                return 0
            else
                return 3
            end
        else
            if state.o
                if state.p || state.v > 1
                    return 0
                else
                    return 3
                end
            else
                if state.p && state.v > 2
                    return 0
                else
                    return 3
                end
            end
        end
    end
end

function save_autonomy_profile(κ)
    # JLD2.save(joinpath(abspath(@__DIR__),"params.jld2"), "κ", κ)
    save_object(joinpath(abspath(@__DIR__),"params.jld2"), κ)
end

function load_autonomy_profile()
    # return load(joinpath(abspath(@__DIR__), "params.jld2"), "κ")
    return load_object(joinpath(abspath(@__DIR__),"params.jld2"))
end

function autonomy_cost(state::CASstate)
    if state.σ == '⊕' || state.σ == '∅'
        return 0.0
    elseif state.σ == '⊖'
        return 3.0 #1.0
    elseif state.σ == '⊘'
        return 10.0 #3.5
    end
end
##

##
mutable struct FeedbackModel
    Σ::Vector{Char}
    λ::Dict{Int, Dict{Int, Dict{Int, Dict{Char, Float64}}}}
    ρ::Function
    D::Dict{String, Dict{String, DataFrame}}
    ϵ::Float64
end

function set_consistency(F::FeedbackModel, ϵ)
    F.ϵ = ϵ
end

function get_state_features(state::DomainState)
    if typeof(state) == NodeState
        return [state.p state.o state.v]
    else
        return [state.o state.l]
    end
end

function generate_feedback_profile(𝒟::DomainSSP,
                                   Σ::Vector{Char},
                                   L::Vector{Int},
                                   D::Dict{String, Dict{String, DataFrame}})
    S, A = 𝒟.S, 𝒟.A
    λ = Dict(s => Dict(a => Dict(l => Dict(σ => 0.5 for σ ∈ Σ)
                                                    for l=1:2)
                                                    for a=1:length(A))
                                                    for s=1:length(S))
    for (a, action) in enumerate(A)
        X_n, Y_n = split_data(D["node"][string(action.value)])
        # M_n = DecisionTreeClassifier(max_depth=8)
        M_n = build_forest(Y_n, X_n, -1, 11, 0.7, -1)
        # DecisionTree.fit!(M_n, X_n, Y_n)
        if action.value ∈ ['↑', '⤉']
            X_e, Y_e = split_data(D["edge"][string(action.value)])
            # M_e = DecisionTreeClassifier(max_depth=8)
            # DecisionTree.fit!(M_e, X_e, Y_e)
            M_e = build_forest(Y_e, X_e, -1, 11, 0.7, -1)
        end

        for (s, state) in enumerate(S)
            if typeof(state) == EdgeState && action.value ∉ ['↑', '⤉']
                continue
            end
            f = get_state_features(state)
            for l in [1,2]
                if typeof(state) == NodeState
                    # pred = predict_proba(M_n, hcat(f,l))
                    pred = apply_forest_proba(M_n, hcat(f,l), [0,1])
                else
                    # pred = predict_proba(M_e, hcat(f,l))
                    pred = apply_forest_proba(M_e, hcat(f,l), [0,1])
                end
                for σ in Σ
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
        X_n, Y_n = split_data(D["node"][string(action.value)])
        # M_n = DecisionTreeClassifier(max_depth=8)
        # DecisionTree.fit!(M_n, X_n, Y_n)
        M_n = build_forest(Y_n, X_n, -1, 11, 0.7, -1)
        if action.value ∈ ['↑', '⤉']
            X_e, Y_e = split_data(D["edge"][string(action.value)])
            # M_e = DecisionTreeClassifier(max_depth=8)
            # DecisionTree.fit!(M_e, X_e, Y_e)
            M_e = build_forest(Y_e, X_e, -1, 11, 0.7, -1)
        end

        for (s, state) in enumerate(S)
            if typeof(state) == EdgeState && action.value ∉ ['↑', '⤉']
                continue
            end
            f = get_state_features(state)
            for l in [1,2]
                if typeof(state) == NodeState
                    # pred = predict_proba(M_n, hcat(f,l))
                    pred = apply_forest_proba(M_n, hcat(f,l), [0,1])
                else
                    # pred = predict_proba(M_e, hcat(f,l))
                    pred = apply_forest_proba(M_e, hcat(f,l), [0,1])
                end
                for σ in Σ
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

# function generate_feedback_profile(𝒟::DomainSSP,
#                                    Σ::Vector{Char},
#                                    L::Vector{Int},
#                                    D::Dict{String, Dict{String, DataFrame}})
#     λ = Dict{Int, Dict{Int, Dict{Int, Dict{Char, Float64}}}}()
#     for (s, state) in enumerate(𝒟.S)
#         f = get_state_features(state)
#         λ[s] = Dict{Int, Dict{Int, Dict{Char, Float64}}}()
#         for (a, action) in enumerate(𝒟.A)
#             λ[s][a] = Dict{Int, Dict{Char, Float64}}()
#             if typeof(state) == NodeState
#                 # X, Y = read_data(joinpath(abspath(@__DIR__), "data", "node_$(action.value).csv"))
#                 X, Y = split_data(D["node"][string(action.value)])
#                 fm = @formula(y ~ x1 + x2 + x3 + x4)
#                 logit = lm(fm, hcat(X, Y), contrasts= Dict(:x3 => DummyCoding()))
#             else
#                 if action.value ∉ ['↑', '⤉']
#                     continue
#                 end
#                 # X, Y = read_data(joinpath(abspath(@__DIR__), "data", "edge_$(action.value).csv"))
#                 X, Y = split_data(D["edge"][string(action.value)])
#                 fm = @formula(y ~ x1 + x2 + x3)
#                 logit = lm(fm, hcat(X, Y), contrasts= Dict(:x2 => DummyCoding()))
#             end
#
#             for l in [1,2]
#                 λ[s][a][l] = Dict{Char, Float64}()
#                 for σ ∈ Σ
#                     q = DataFrame(hcat(f, l), :auto)
#                     p = clamp(predict(logit, q)[1], 0.0, 1.0)
#                     if σ == '⊕' || σ == '∅'
#                         λ[s][a][l][σ] = p
#                     else
#                         λ[s][a][l][σ] = 1.0 - p
#                     end
#                 end
#             end
#         end
#     end
#     return λ
# end

# function update_feedback_profile!(C)
#     λ, 𝒟, Σ, L, D = C.𝒮.F.λ, C.𝒮.D, C.𝒮.F.Σ, C.𝒮.A.L, C.𝒮.F.D
#     for (s, state) in enumerate(𝒟.S)
#         f = get_state_features(state)
#         for (a, action) in enumerate(𝒟.A)
#             if typeof(state) == NodeState
#                 # X, Y = read_data(joinpath(abspath(@__DIR__), "data", "node_$(action.value).csv"))
#                 X, Y = split_data(D["node"][string(action.value)])
#                 fm = @formula(y ~ x1 + x2 + x3 + x4)
#                 logit = lm(fm, hcat(X, Y), contrasts= Dict(:x3 => DummyCoding()))
#             else
#                 if action.value ∉ ['↑', '⤉']
#                     continue
#                 end
#                 # X, Y = read_data(joinpath(abspath(@__DIR__), "data", "edge_$(action.value).csv"))
#                 X, Y = split_data(D["edge"][string(action.value)])
#                 fm = @formula(y ~ x1 + x2 + x3)
#                 logit = lm(fm, hcat(X, Y), contrasts= Dict(:x2 => DummyCoding()))
#             end
#             for l in [1,2]
#                 for σ ∈ Σ
#                     q = DataFrame(hcat(f, l), :auto)
#                     p = clamp(predict(logit, q)[1], 0.0, 1.0)
#                     if σ == '⊕' || σ == '∅'
#                         λ[s][a][l][σ] = p
#                     else
#                         λ[s][a][l][σ] = 1.0 - p
#                     end
#                 end
#             end
#         end
#     end
# end

function save_feedback_profile(λ)
    # JLD2.save(joinpath(abspath(@__DIR__),"params.jld2"), "λ", λ)
    save_object(joinpath(abspath(@__DIR__),"params.jld2"), λ)
end

function load_feedback_profile()
    # return load(joinpath(abspath(@__DIR__), "params.jld2", "λ"))
    return load_object(joinpath(abspath(@__DIR__),"params.jld2"))
end

function save_data(D)
    for k in keys(D["edge"])
        record_data(D["edge"][k], joinpath(abspath(@__DIR__), "data", "edge_$k.csv"), false)
    end
    for k in keys(D["node"])
        record_data(D["node"][k], joinpath(abspath(@__DIR__), "data", "node_$k.csv"), false)
    end
end

function human_cost(action::CASaction)
    return [10.0 2.0 1.0 0.][action.l + 1]#[5. 1.5 .5 0.][action.l + 1]              #TODO: Fix this.
end
##

struct CAS
    D::DomainSSP
    A::AutonomyModel
    F::FeedbackModel
end

mutable struct CASSP
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
    s_length, a_length = size(𝒮.D.S)[1], size(𝒮.D.A)[1]
    flags = Dict(s => Dict(a => false for a=1:a_length) for s=1:s_length)
    potential = Dict(s => Dict(a => [0. for i=1:4] for a=1:a_length) for s=1:s_length)
    return CASSP(𝒮, S, A, T, C, s₀, G, SIndex, AIndex, flags, potential)
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

function allowed(C, s::Int,
                    a::Int)
    return C.A[a].l <= C.𝒮.A.κ[ceil(s/4)][ceil(a/4)]
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
            if t == [(base_s, 1.0)]  || action.l > κ[base_s][base_a]
                T[s][a] = [(s, 1.0)]
                continue
            end

            T[s][a] = Vector{Tuple{Int, Float64}}()
            if action.l == 0
                if typeof(state.state) == EdgeState
                    state′ = CASstate(EdgeState(state.state.u, state.state.v,
                                state.state.θ, false, state.state.l), '∅')
                    T[s][a] = [(C.SIndex[state′], 1.0)]
                else
                    T[s][a] = [((t[argmax([x[2] for x in t])][1]-1) * 4 + 4 , 1.0)]
                end
            elseif action.l == 1
                p_approve = λ[base_s][base_a][1]['⊕']
                p_disapprove = 1.0 - p_approve #λ[base_s][base_a][1]['⊖']
                push!(T[s][a], ((base_s-1) * 4 + 2, p_disapprove))
                for (sp, p) in t
                    push!(T[s][a], ((sp-1) * 4 + 1, p * p_approve))
                end
            elseif action.l == 2
                p_override = λ[base_s][base_a][2]['⊘']
                p_null = 1.0 - p_override
                append!(T[s][a], ( (x-1, y * p_override) for (x,y) in T[s][C.AIndex[CASaction(action.action, 0)]]))
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

function generate_feedback(state::CASstate,
                          action::CASaction,
                               ϵ::Float64)

    if typeof(state.state) == EdgeState && !state.state.o && action.action.value == '↑'
      return (action.l == 1) ? '⊕' : '∅'
    end

    if rand() < 1-ϵ
        if action.l == 1
            return ['⊕', '⊖'][rand(1:2)]
        elseif action.l == 2
            return ['∅', '⊘'][rand(1:2)]
        end
    end

    if typeof(state.state) == EdgeState
        if state.state.o && state.state.l == 1
            return (action.l == 1) ? '⊖' : '⊘'
        else
            return (action.l == 1) ? '⊕' : '∅'
        end
    else
        if action.action.value == '⤉'
            return (action.l == 1) ? '⊕' : '∅'
        elseif action.action.value == '→'
            if state.state.o && state.state.p && state.state.v > 1
                return (action.l == 1) ? '⊖' : '⊘'
            else
                return (action.l == 1) ? '⊕' : '∅'
            end
        else
            if state.state.o
                if state.state.p || state.state.v > 1
                    return (action.l == 1) ? '⊖' : '⊘'
                else
                    return (action.l == 1) ? '⊕' : '∅'
                end
            else
                if state.state.p && state.state.v > 2
                    return (action.l == 1) ? '⊖' : '⊘'
                else
                    return (action.l == 1) ? '⊕' : '∅'
                end
            end
        end
    end
end

function generate_successor(M::DomainSSP,
                        state::CASstate,
                       action::CASaction,
                            σ::Char)
    s, a = M.SIndex[state.state], M.AIndex[action.action]
    thresh = rand()
    p = 0.
    T = M.T[s][a]
    for (s′, prob) ∈ T
        p += prob
        if p >= thresh
            return CASstate(M.S[s′], σ)
        end
    end
end

function reachable(C, L)
    s, S = C.SIndex[C.s₀], C.S
    reachable = Set{Int}()
    to_visit = Vector{Int}()
    push!(to_visit, s)
    while !isempty(to_visit)
        if terminal(C, C.S[s])
            s = pop!(to_visit)
            continue
        end
        a = solve(L, C, s)[1]
        for (sp, p) in C.T[s][a]
            if sp ∉ reachable && p > 0.0
                push!(to_visit, sp)
                push!(reachable, sp)
            end
        end
        s = pop!(to_visit)
    end
    return reachable
end

function compute_level_optimality(C, ℒ)
    total = 0
    r = 0
    lo = 0
    lo_r = 0
    # for s in keys(ℒ.π)
    R = reachable(C, ℒ)
    for (s, state) in enumerate(C.S)
        if terminal(C, state)
            continue
        end
        solve(ℒ, C, s)
        total += 1
        # state = C.S[s]
        action = C.A[ℒ.π[s]]
        comp = (action.l == competence(state.state, action.action))
        lo += comp
        if s in R
            r += 1
            lo_r += comp
        end
    end
    # println("  ")
    # println(lo)
    # println(total)

    return lo/total, lo_r/r
end

function build_cas(𝒟::DomainSSP,
                   L::Vector{Int},
                   Σ::Vector{Char})
    if ispath(joinpath(abspath(@__DIR__), "params.jld"))
        κ = load_autonomy_profile()
    else
        κ = generate_autonomy_profile(𝒟)
    end
    𝒜 = AutonomyModel(L, κ, autonomy_cost)

    D = Dict{String, Dict{String, DataFrame}}()

    D["node"] = Dict{String, DataFrame}()
    for a in ["↑", "→", "↓", "←", "⤉"]
        D["node"][a] = DataFrame(CSV.File(joinpath(abspath(@__DIR__), "data", "node_$a.csv")))
    end
    D["edge"] = Dict{String, DataFrame}()
    for a in ["↑", "⤉"]
        D["edge"][a] = DataFrame(CSV.File(joinpath(abspath(@__DIR__), "data", "edge_$a.csv")))
    end
    λ = generate_feedback_profile(𝒟, Σ, L, D)
    ℱ = FeedbackModel(Σ, λ, human_cost, D, 0.9)
    𝒮 = CAS(𝒟, 𝒜, ℱ)
    S, s₀, G = generate_states(𝒟, ℱ)
    A = generate_actions(𝒟, 𝒜)
    T = Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}}()

    C = CASSP(𝒮, S, A, T, generate_costs, s₀, G)
    generate_transitions!(𝒟, 𝒜, ℱ, C, S, A, G)
    check_transition_validity(C)
    return C
end

function solve_model(C::CASSP)
    ℒ = LRTDPsolver(C, 10000., 100, .001, Dict{Int, Int}(),
                     false, Set{Int}(), zeros(length(C.S)),
                                        zeros(length(C.A)))
    solve(ℒ, C, C.SIndex[C.s₀])
    return ℒ
end

function init_data()
    for action in ["←", "↑", "→", "↓", "⤉"]
        init_node_data(joinpath(abspath(@__DIR__), "data", "node_$action.csv"))
    end

    init_edge_data(joinpath(abspath(@__DIR__), "data", "edge_↑.csv"))
    init_edge_data(joinpath(abspath(@__DIR__), "data", "edge_⤉.csv"))
end

function set_route(M, C, init, goal)
    set_init!(M, init)
    set_goal!(M, goal)
    generate_transitions!(M, M.graph)
    reset_problem!(M, C)
end

function random_route(M, C)
    init = rand([12, 1, 4, 16])
    goal = rand(5:8)
    while goal == init
        goal = rand(1:16)
    end
    set_init!(M, init)
    set_goal!(M, goal)
    generate_transitions!(M, M.graph)
    reset_problem!(M, C)
end


function get_route(M, C, L)
    route = Vector{Int}()
    state = C.s₀
    while !(state ∈ C.G)
        if typeof(state.state) == NodeState && (isempty(route) || last(route) != state.state.id)
            push!(route, state.state.id)
        end
        s = C.SIndex[state]
        # a = L.π[s]
        a = solve(L, C, s)[1]
        # println(state,  "     |     ", C.A[a])
        state = generate_successor(M, state, C.A[a], '∅')
        # sp = C.T[s][a][1][1]
        # state = C.S[sp]
    end
    push!(route, state.state.id)
    return route
end

function debug_competence(C, L)
    κ, λ, D = C.𝒮.A.κ, C.𝒮.F.λ, C.𝒮.D
    for (s, state) in enumerate(C.S)
        # println("**** $s ****")
        state = C.S[s]
        if terminal(C, state)
            continue
        end
        ds = Int(ceil(s/4))
        a = solve(L, C, s)[1]
        action = C.A[a]
        da = Int(ceil(a/4))
        if action.l != competence(state.state, action.action)
            println("-----------------------")
            println("State:  $state      $s |       Action: $action         $a")
            println("Competence: $(competence(state.state, action.action))")
            println("Kappa: $(κ[ds][da])")
            println("Lambda: $(λ[ds][da])")
            println("-----------------------")
        end
    end
end
# debug_competence(C, L)
