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
using DecisionTree
using DataFrames
using CSV
using JLD2
using StatsBase

include("domain_model.jl")
struct CASstate
    state::DomainState
        σ::Char
end
function ==(a::CASstate, b::CASstate)
    return isequal(a.state, b.state) && isequal(a.σ, b.σ)
end
function Base.hash(a::CASstate, h::UInt)
    h = hash(a.state, h)
    h = hash(a.σ, h)
    return h
end

struct CASaction
    action::DomainAction
         l::Int
end
function ==(a::CASaction, b::CASaction)
    return isequal(a.action, b.action) && isequal(a.l, b.l)
end
function Base.hash(a::CASaction, h::UInt)
    h = hash(a.action, h)
    h = hash(a.l, h)
    return h
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
            κ[s][a] = competence(state, action)
            # if typeof(state) == EdgeState && action.value == '↑'
            #     if state.r == "None"
            #         κ[s][a] = 2
            #     else
            #         κ[s][a] = 0
            #     end
            # else
            #     if typeof(state) == NodeState && action.value == '⤉'
            #         κ[s][a] = 2
            #     elseif typeof(state) == NodeState && (!state.o && !state.p && state.v == 0)
            #         κ[s][a] = 2
            #     else
            #         κ[s][a] = 1
            #     end
            # end
        end
    end
    return κ
end

function update_potential(C, ℒ, s, a, L)
    state = CASstate(C.𝒮.D.S[s], '∅')
    s2 = C.SIndex[state]
    X = [lookahead(ℒ, s2, ((a - 1) * 3 + l + 1) ) for l ∈ L]
    P = softmax(-1.0 .* X)
    for l=1:size(L)[1]
        C.potential[s][a][L[l]+1] += P[l]
    end
    clamp!(C.potential[s][a], 0.0, 1.0)
end

function update_autonomy_profile!(C, ℒ)
    κ = C.𝒮.A.κ
    for (s, state) in enumerate(C.𝒮.D.S)
        if state.w != C.s₀.state.w
            continue
        end
        for (a, action) in enumerate(C.𝒮.D.A)
            if κ[s][a] == competence(state, action)
                continue
            end
            if typeof(state) == EdgeState && action.value ∉ ['↑', '⤉']
                continue
            end

            L = [0,1,2]
            update_potential(C, ℒ, s, a, L)
            distr = softmax([C.potential[s][a][l+1] for l in L])
            i = sample(aweights(distr))

            if L[i] == 2
                if C.𝒮.F.λ[s][a][1]['∅'] < 0.85
                    C.potential[s][a][L[i] + 1] = 0.0
                    continue
                end
            elseif L[i] == 0
                if C.𝒮.F.λ[s][a][0]['⊕'] > 0.25
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
        end
    end
end

function competence(state::DomainState,
                   action::DomainAction)
    if typeof(state) == EdgeState
        if action.value == '↑'
            if state.r == "None"
                return 2
            else
                return 0
            end
        elseif action.value == '⤉'
            if state.o && state.l == 1
                return 0
            else
                return 2
            end
        else
            return 2
        end
    else
        if action.value == '⤉'
            return 2
        elseif action.value == '→'
            if (state.o || state.w.weather == "snowy" || (state.w.time == "night" && state.w.weather == "rainy")) && state.p && state.v > 1
                return 0
            else
                return 2
            end
        else
            if state.o || state.w.weather == "snowy" || (state.w.time == "night" && state.w.weather == "rainy")
                if state.p || state.v > 1
                    return 0
                else
                    return 2
                end
            else
                if state.p && state.v > 2
                    return 0
                else
                    return 2
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
    if state.σ == '⊕'
        return 0.0
    elseif state.σ == '∅'
        if state.state.w.active_avs == 1
            return 2.0 * 4.0
        else
            return 2.0 * (state.state.w.active_avs-1.0)
        end #1.0
    else
        return 2.0
    end
end
##

##
# mutable struct OperatorModel
#    SH::Set{Vector{Int}} # Operator state vector length n+1
#    TH::Function
#     Σ::Vector{Char}
#     λ::Dict{Int, Dict{Int, Dict{Int, Dict{Char, Float64}}}}
#     ρ::Function
#     D::Dict{String, Dict{String, DataFrame}}
#     ϵ::Float64
# end

# function generate_random_operator_state()
#     o1, o2 = sample(1:2), sample(1:2)
#     oa = (o1 == 1) ? 1 : 2
#     return [o1, o2, oa]
# end

# function human_state_transition(sh, s, a, l)
#     o1, o2, oa = sh[1], sh[2], sh[3]
#
#     T = Vector{Tuple{Vector, Float32}}()
#     if o1 == 1 # Local operator available --> state is [1, x, 1]
#         # Local operator becomes busy (only happens if not using operator)
#         p_becomes_busy = 1.0 - (0.5)^s.w.active_avs
#
#         if o2 == 1
#             push!(T, ([2, 1, 2], p_becomes_busy * 0.75))
#             push!(T, ([2, 2, 2], p_becomes_busy * 0.25))
#             push!(T, ([1, 1, 1], (1.0-p_becomes_busy) * 0.75))
#             push!(T, ([1, 2, 1], (1.0-p_becomes_busy) * 0.25))
#         else
#             push!(T, ([2, 1, 2], p_becomes_busy * 0.25))
#             push!(T, ([2, 2, 2], p_becomes_busy * 0.75))
#             push!(T, ([1, 1, 1], (1.0-p_becomes_busy) * 0.25))
#             push!(T, ([1, 2, 1], (1.0-p_becomes_busy) * 0.75))
#         end
#
#     else # Local operator unavailable --> state is [2, x, 2]
#         p_becomes_active = (0.5)^s.w.active_avs
#         if o2 == 1
#             push!(T, ([1, 1, 1], p_becomes_active * 0.75))
#             push!(T, ([1, 2, 1], p_becomes_active * 0.25))
#             push!(T, ([2, 1, 2], (1.0 - p_becomes_active) * 0.75))
#             push!(T, ([2, 2, 2], (1.0 - p_becomes_active) * 0.25))
#         else
#             push!(T, ([1, 1, 1], p_becomes_active * 0.25))
#             push!(T, ([1, 2, 1], p_becomes_active * 0.75))
#             push!(T, ([2, 1, 2], (1.0 - p_becomes_active) * 0.25))
#             push!(T, ([2, 2, 2], (1.0 - p_becomes_active) * 0.75))
#         end
#     end
#     return T
# end

# function human_state_transition(sh, s, a, l)
#     o1, o2, oa = sh[1], sh[2], sh[3]
#
#     T = Vector{Tuple{Vector, Float32}}()
#     if o1 == 1 # Local operator available --> state is [1, x, 1]
#         if l == 2
#             # Local operator becomes busy (only happens if not using operator)
#             p_becomes_busy = 1.0 - (0.5)^s.w.active_avs
#             # Global operator takes over.
#             if o2 == 1
#                 push!(T, ([2, 1, 2], p_becomes_busy * 0.75))
#                 push!(T, ([2, 2, 2], p_becomes_busy * 0.25))
#                 push!(T, ([1, 1, 1], (1.0-p_becomes_busy) * 0.75))
#                 push!(T, ([1, 2, 1], (1.0-p_becomes_busy) * 0.25))
#             else
#                 push!(T, ([2, 1, 2], p_becomes_busy * 0.25))
#                 push!(T, ([2, 2, 2], p_becomes_busy * 0.75))
#                 push!(T, ([1, 1, 1], (1.0-p_becomes_busy) * 0.25))
#                 push!(T, ([1, 2, 1], (1.0-p_becomes_busy) * 0.75))
#             end
#         else
#             if o2 == 1
#                 push!(T, ([1, 1, 1], 0.75))
#                 push!(T, ([1, 2, 1], 0.25))
#             else
#                 push!(T, ([1, 1, 1], 0.25))
#                 push!(T, ([1, 2, 1], 0.75))
#             end
#         end
#     else # Local operator unavailable --> state is [2, x, 2]
#         p_becomes_active = (0.5)^s.w.active_avs
#         if o2 == 1
#             push!(T, ([1, 1, 1], p_becomes_active * 0.75))
#             push!(T, ([1, 2, 1], p_becomes_active * 0.25))
#             push!(T, ([2, 1, 2], (1.0 - p_becomes_active) * 0.75))
#             push!(T, ([2, 2, 2], (1.0 - p_becomes_active) * 0.25))
#         else
#             push!(T, ([1, 1, 1], p_becomes_active * 0.25))
#             push!(T, ([1, 2, 1], p_becomes_active * 0.75))
#             push!(T, ([2, 1, 2], (1.0 - p_becomes_active) * 0.25))
#             push!(T, ([2, 2, 2], (1.0 - p_becomes_active) * 0.75))
#         end
#     end
#     return T
# end

# function get_consistency(sh)
#     o1, o2, oa = sh[1],sh[2],sh[3]
#     if oa == 1
#         return 1.0
#     else
#         if o2 == 1
#             return 0.8
#         else
#             return 0.7
#         end
#     end
# end

# function set_consistency(F::OperatorModel, ϵ)
#     F.ϵ = ϵ
# end

# function get_state_features(state::DomainState)
#     if typeof(state) == NodeState
#         return [state.p state.o state.v state.w.active_avs state.w.time state.w.weather]
#     else
#         return [state.o state.l state.w.active_avs state.w.time state.w.weather]
#     end
# end

function generate_cas_feedback_profile(𝒟::DomainSSP,
                                       Σ::Vector{Char},
                                       L::Vector{Int},
                                       D::Dict{String, Dict{String, DataFrame}})
    S, A = 𝒟.S, 𝒟.A
    λ = Dict(s => Dict(a => Dict(l => Dict(σ => 0.5 for σ ∈ Σ)
                                                    for l=0:1)
                                                    for a=1:length(A))
                                                    for s=1:length(S))

    Threads.@threads for s=1:length(S)
        state = S[s]
        for a=1:length(A)
            action = A[a]
            for l=0:1
                σ = generate_feedback(COCASstate([2,1,2],state,'∅'), COCASaction(action,l), 1.0)
                if σ == '⊕'
                    λ[s][a][l]['⊕'] = .8333
                    λ[s][a][l]['⊖'] = 1. - .8333
                elseif σ == '⊖'
                    λ[s][a][l]['⊕'] = 1. - .8333
                    λ[s][a][l]['⊖'] = .8333
                elseif σ == '∅'
                    λ[s][a][l]['∅'] = .8333
                    λ[s][a][l]['⊘'] = 1. - .8333
                else
                    λ[s][a][l]['∅'] = 0.
                    λ[s][a][l]['⊘'] = 1.
                end
            end
        end
    end
    # for (a, action) in enumerate(A)
    #     X_n, Y_n = split_data(D["node"][string(action.value)])
    #     M_n = build_forest(Y_n, X_n, -1, 11, 0.7, -1)
    #     if action.value ∈ ['↑', '⤉']
    #         X_e, Y_e = split_data(D["edge"][string(action.value)])
    #         M_e = build_forest(Y_e, X_e, -1, 11, 0.7, -1)
    #     end
    #
    #     for (s, state) in enumerate(S)
    #         if typeof(state) == EdgeState && action.value ∉ ['↑', '⤉']
    #             continue
    #         end
    #         f = get_state_features(state)
    #         for l in [0,1]
    #             if typeof(state) == NodeState
    #                 pred = apply_forest_proba(M_n, hcat(f,l), [0,1])
    #             else
    #                 pred = apply_forest_proba(M_e, hcat(f,l), [0,1])
    #             end
    #             for σ in Σ
    #                 if σ == '⊖' || σ == '⊘'
    #                     λ[s][a][l][σ] = pred[1]
    #                 else
    #                     try
    #                         λ[s][a][l][σ] = pred[2]
    #                     catch
    #                         print(s, "|", a, "|", l)
    #                     end
    #                 end
    #             end
    #         end
    #     end
    # end
    return λ
end

# function update_feedback_profile!(C)
#     λ, 𝒟, Σ, L, D = C.𝒮.F.λ, C.𝒮.D, C.𝒮.F.Σ, C.𝒮.A.L, C.𝒮.F.D
#     S, A = 𝒟.S, 𝒟.A
#     for (a, action) in enumerate(A)
#         X_n, Y_n, M_n = missing, missing, missing
#         failed_to_build_node, failed_to_build_edge = false, false
#         try
#             X_n, Y_n = split_data(D["node"][string(action.value)])
#             M_n = build_forest(Y_n, X_n, -1, 11, 0.7, -1)
#         catch
#             failed_to_build_node = true
#         end
#
#         X_e, Y_e, M_e = missing, missing, missing
#         if action.value ∈ ['↑', '⤉']
#             try
#                 X_e, Y_e = split_data(D["edge"][string(action.value)])
#                 M_e = build_forest(Y_e, X_e, -1, 11, 0.7, -1)
#             catch
#                 failed_to_build_edge = true
#             end
#         end
#
#         for (s, state) in enumerate(S)
#             if typeof(state) == EdgeState && action.value ∉ ['↑', '⤉']
#                 continue
#             end
#             f = get_state_features(state)
#             for l in [0,1]
#                 if typeof(state) == NodeState
#                     if failed_to_build_node
#                         for σ ∈ Σ
#                             λ[s][a][l][σ] = 0.5
#                         end
#                         continue
#                     else
#                         pred = []
#                         try
#                             pred = apply_forest_proba(M_n, hcat(f,l), [0,1])
#                         catch
#                             pred = [0.5 0.5]
#                         end
#                         for σ in Σ
#                             if σ == '⊖' || σ == '⊘'
#                                 λ[s][a][l][σ] = pred[1]
#                             else
#                                 λ[s][a][l][σ] = pred[2]
#                             end
#                         end
#                     end
#                 else
#                     if failed_to_build_edge
#                         for σ ∈ Σ
#                             λ[s][a][l][σ] = 0.5
#                         end
#                         continue
#                     else
#                         pred = []
#                         try
#                             pred = apply_forest_proba(M_e, hcat(f,l), [0,1])
#                         catch
#                             pred = [0.5 0.5]
#                         end
#                         for σ in Σ
#                             if σ == '⊖' || σ == '⊘'
#                                 λ[s][a][l][σ] = pred[1]
#                             else
#                                 λ[s][a][l][σ] = pred[2]
#                             end
#                         end
#                     end
#                 end
#             end
#         end
#     end
# end

# function save_feedback_profile(λ)
#     save_object(joinpath(abspath(@__DIR__),"params.jld2"), λ)
# end
#
# function load_feedback_profile()
#     return load_object(joinpath(abspath(@__DIR__),"params.jld2"))
# end

# function save_data(D)
#     for k in keys(D["edge"])
#         record_data(D["edge"][k], joinpath(abspath(@__DIR__), "data", "edge_$k.csv"), false)
#     end
#     for k in keys(D["node"])
#         record_data(D["node"][k], joinpath(abspath(@__DIR__), "data", "node_$k.csv"), false)
#     end
# end

function human_cost(action::CASaction)
    return [1.0 1.0 0.0][action.l + 1]
end
##

struct CAS
    D::DomainSSP
    A::AutonomyModel
    F::OperatorModel
end

mutable struct CASSP
    𝒮::CAS
    S::Vector{CASstate}
    A::Vector{CASaction}
    T
    C::Array{Array{Float64,1},1}
   s₀::CASstate
    G::Set{CASstate}
    SIndex::Dict{CASstate, Int}
    AIndex::Dict{CASaction, Int}
    potential::Dict{Int, Dict{Int, Vector{Float64}}}
end
function CASSP(𝒮::CAS,
               S::Vector{CASstate},
               A::Vector{CASaction},
               T::Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}},
               C::Array{Array{Float64,1},1},
              s₀::CASstate,
               G::Set{CASstate})
    SIndex, AIndex = generate_index_dicts(S, A)
    s_length, a_length = size(𝒮.D.S)[1], size(𝒮.D.A)[1]
    potential = Dict(s => Dict(a => [0. for i=1:3] for a=1:a_length) for s=1:s_length)
    return CASSP(𝒮, S, A, T, C, s₀, G, SIndex, AIndex, potential)
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
            if state in D.G && σ == '⊕'
                push!(G, new_state)
            end
        end
    end
    return states, CASstate(D.s₀, '⊕'), G
end

function reset_problem!(D, C)
    C.s₀ = CASstate(D.s₀, '⊕')
    C.G = Set{CASstate}()
    for state in D.G
        push!(C.G, CASstate(state, '⊕'))
    end
    generate_costs!(D)
    generate_costs!(C)
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
    return C.A[a].l <= C.𝒮.A.κ[C.𝒮.D.SIndex[C.S[s].state]][Int(ceil(a/3))]
end

function generate_transitions!(𝒟, 𝒜, ℱ, C,
                              S::Vector{CASstate},
                              A::Vector{CASaction},
                              G::Set{CASstate})

    T = C.T
    κ, λ = 𝒜.κ, ℱ.λ
    for (s, state) in enumerate(S)
        if state.state.w.time != C.s₀.state.w.time || state.state.w.weather != C.s₀.state.w.weather
            continue
        end
        T[s] = Dict{Int, Vector{Tuple{Int, Float64}}}()
        for (a, action) in enumerate(A)
            if state.state in 𝒟.G
                state′ = CASstate(state.state, '⊕')
                T[s][a] = [(C.SIndex[state′], 1.0)]
                continue
            end

            base_state = state.state
            base_action = action.action
            base_s = 𝒟.SIndex[base_state]
            base_a = 𝒟.AIndex[base_action]
            w = state.state.w
            if w.active_avs == 4
                w = WorldState(1, w.time, w.weather)
            else
                w = WorldState(w.active_avs+1, w.time, w.weather)
            end
            t = 𝒟.T[base_s][base_a]
            if (t == [(base_s, 1.0)]  || action.l > κ[base_s][base_a])
                T[s][a] = Vector{Tuple{Int, Float64}}()
                if typeof(state.state) == NodeState
                    dstate′ = NodeState(state.state.id, state.state.p,
                        state.state.o, state.state.v, state.state.θ, w)
                else
                    dstate′ = EdgeState(state.state.u, state.state.v,
                        state.state.θ, state.state.o, state.state.l,
                        state.state.r, w)
                end
                push!(T[s][a], (C.SIndex[CASstate(dstate′, state.σ)], 1.0))
                continue
            end

            T[s][a] = Vector{Tuple{Int, Float64}}()
            if action.l == 0
                p_approval = λ[base_s][base_a][0]['∅']
                p_disapproval = λ[base_s][base_a][0]['⊘']

                if typeof(state.state) == EdgeState
                    if state.state.o && action.action.value == '⤉'
                        dstate′ = EdgeState(state.state.u,
                                state.state.v, state.state.θ, false,
                                state.state.l, state.state.r, w)
                        state′ = CASstate(dstate′, '∅')
                        push!(T[s][a], (C.SIndex[state′], p_approval))
                        dstate′′ = EdgeState(state.state.u,
                                state.state.v, state.state.θ, true,
                                state.state.l, state.state.r, state.state.w)
                        push!(T[s][a], (C.SIndex[CASstate(dstate′′, '⊘')], p_disapproval))
                    elseif !state.state.o && action.action.value == '↑'
                        temp = []
                        mass = 0.0
                        for j=1:length(t)
                            if typeof(𝒟.S[t[j][1]]) == NodeState
                                push!(temp, t[j])
                                mass += t[j][2]
                            end
                        end
                        for j=1:length(temp)
                            state′ = CASstate(𝒟.S[temp[j][1]], '∅')
                            push!(T[s][a], (C.SIndex[state′], (temp[j][2]/mass)*p_approval))
                        end
                        dstate′ = EdgeState(state.state.u, state.state.v,
                                state.state.θ, state.state.o, state.state.l,
                                state.state.r, w)
                        push!(T[s][a], (C.SIndex[CASstate(dstate′, '⊘')], p_disapproval))
                    else
                        dstate′ = EdgeState(state.state.u, state.state.v,
                                state.state.θ, state.state.o, state.state.l,
                                state.state.r, w)
                        T[s][a] = [(C.SIndex[CASstate(dstate′, state.σ)], 1.0)]
                        continue
                    end
                else
                    for j = 1:length(t)
                        push!(T[s][a], (C.SIndex[CASstate(𝒟.S[t[j][1]], '∅')],
                                        t[j][2] * p_approval))
                    end
                    dstate′ = NodeState(state.state.id, state.state.p,
                        state.state.o, state.state.v, state.state.θ, w)
                    push!(T[s][a], (C.SIndex[CASstate(dstate′, '⊘')], p_disapproval))
                end
            elseif action.l == 1
                p_approve = λ[base_s][base_a][1]['⊕']
                p_disapprove = 1.0 - p_approve #λ[base_s][base_a][1]['⊖']
                if typeof(state.state) == NodeState
                    dstate′ = NodeState(state.state.id, state.state.p,
                        state.state.o, state.state.v, state.state.θ, w)
                else
                    dstate′ = EdgeState(state.state.u, state.state.v,
                        state.state.θ, state.state.o, state.state.l,
                        state.state.r, w)
                end
                push!(T[s][a], (C.SIndex[CASstate(dstate′, '⊖')],
                                p_disapprove))
                for j=1:length(t)
                    push!(T[s][a], (C.SIndex[CASstate(𝒟.S[t[j][1]], '⊕')],
                                t[j][2] * p_approve))
                end
            else
                for j=1:length(t)
                    push!(T[s][a], (C.SIndex[CASstate(
                            𝒟.S[t[j][1]], '⊕')], t[j][2]))
                end
            end
        end
    end
end

function check_transition_validity(C)
    S, A, T = C.S, C.A, C.T
    for s in keys(T)
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
    # TODO: why do we not block C.T[s][a] as well? Not understanding...
    C.T[s][a] = [(s, 1.0)]
    C.T[s+1][a] = [(s+1, 1.0)]
    C.T[s+2][a] = [(s+2, 1.0)]
    C.T[s+3][a] = [(s+3, 1.0)]
end

function generate_costs(C::CASSP,
                        s::Int,
                        a::Int,)
    D, A, F = C.𝒮.D, C.𝒮.A, C.𝒮.F
    state, action = C.S[s], C.A[a]
    cost = D.C[D.SIndex[state.state]][D.AIndex[action.action]]
    cost += A.μ(state)
    cost += F.ρ(action)
    return cost
end

function generate_costs!(C::CASSP)
    Threads.@threads for s = 1:length(C.S)
        for a = 1:length(C.A)
            C.C[s][a] = generate_costs(C, s, a)
        end
    end
    # C.C = [[generate_costs(C, s, a) for a=1:length(C.A)] for s=1:length(C.S)]
end

function generate_feedback(state::CASstate,
                          action::CASaction,
                              sh,
                               ϵ::Float64)
    # Request for ToC logic
    if action.l == 0
        # Operator noise
        if rand() < 1 - get_consistency(sh)
            return ['⊘', '∅'][rand(1:2)]
        end

        if sh[3] == 1 # Local operator always accepts
            return '∅'
        else
            if sh[2] == 1
                # Global operator accepts if:
                #   - No pedestrians
                #   - No occlusions
                #   - < 4 vehicle complexity
                #   - daytime and sunny
                if state.state.w.time == "day" && state.state.w.weather == "sunny"
                    if (typeof(state.state) == EdgeState) || (!state.state.p && !state.state.o && state.state.v < 3)
                        return '∅'
                    end
                    # if rand() < (1 - get_cosistency(state.sh))/2
                    #     return '⊘'
                    # else
                    #     return '∅'
                    # end
                    return '⊘'
                else
                    return '⊘'
                end
            else # Global operator always rejects with bad connection
                return '⊘'
            end
        end
    end

    # if typeof(state.state) == EdgeState && !state.state.o && action.action.value == '↑'
    #   return (action.l == 1) ? '⊕' : '∅'
    # end
    #
    # if rand() < 1 - get_consistency(sh)
    #     return ['⊕', '⊖'][rand(1:2)]
    # end
    #
    # if sh[3] == 2
    #     if sh[2] == 1
    #         if (state.state.w.time == "night" && state.state.w.weather == "snowy")
    #             return (action.l == 1) ? '⊖' : '⊘'
    #         end
    #     else
    #         if (state.state.w.time == "night" && state.state.w.weather == "rainy" ||
    #             state.state.w.weather == "snowy")
    #             return (action.l == 1) ? '⊖' : '⊘'
    #         end
    #     end
    # end
    #
    # if typeof(state.state) == EdgeState
    #     if state.state.o && state.state.l == 1
    #         return (action.l == 1) ? '⊖' : '⊘'
    #     else
    #         return (action.l == 1) ? '⊕' : '∅'
    #     end
    # else
    #     if action.action.value == '⤉'
    #         return (action.l == 1) ? '⊕' : '∅'
    #     elseif action.action.value == '→'
    #         if (state.o || state.w.weather == "snowy" || (state.w.time == "night" && state.w.weather == "rainy")) && state.p && state.v > 1
    #             return (action.l == 1) ? '⊖' : '⊘'
    #         else
    #             return (action.l == 1) ? '⊕' : '∅'
    #         end
    #     else
    #         if state.o || state.w.weather == "snowy" || (state.w.time == "night" && state.w.weather == "rainy")
    #             if state.state.p || state.state.v > 1
    #                 return (action.l == 1) ? '⊖' : '⊘'
    #             else
    #                 return (action.l == 1) ? '⊕' : '∅'
    #             end
    #         else
    #             if state.state.p && state.state.v > 2
    #                 return (action.l == 1) ? '⊖' : '⊘'
    #             else
    #                 return (action.l == 1) ? '⊕' : '∅'
    #             end
    #         end
    #     end
    # end
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
        if terminal(C, state) || state.state.w != C.s₀.state.w
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

    κ = generate_autonomy_profile(𝒟)
    𝒜 = AutonomyModel(L, κ, autonomy_cost)

    D = Dict{String, Dict{String, DataFrame}}()
    λ = generate_cas_feedback_profile(𝒟, Σ, L, D)

    SH = Set([i for i in x] for x in vec(collect(Base.product(1:2, 1:2, 1:2))))
    ℱ = OperatorModel(SH, human_state_transition, Σ, λ, human_cost, D, 0.9)
    𝒮 = CAS(𝒟, 𝒜, ℱ)
    S, s₀, G = generate_states(𝒟, ℱ)
    A = generate_actions(𝒟, 𝒜)
    T = Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}}()
    costs = [[0. for a=1:length(A)] for s=1:length(S)]
    C = CASSP(𝒮, S, A, T, costs, s₀, G)
    generate_costs!(C)
    generate_transitions!(𝒟, 𝒜, ℱ, C, S, A, G)
    check_transition_validity(C)
    return C
end

function solve_model(C::CASSP)
    L = solve_model(C.𝒮.D)
    ℒ = LRTDPsolver(C, 10000., 1000, .01, Dict{Int, Int}(),
                     false, Set{Int}(), L.V, zeros(length(C.S)),
                                             zeros(length(C.A)))
    solve(ℒ, C, C.SIndex[C.s₀])
    return ℒ
end

function init_data()
    for action in ["←", "↑", "→", "↓", "⤉"]
        init_cas_node_data(joinpath(abspath(@__DIR__), "data", "node_$action.csv"))
    end

    init_cas_edge_data(joinpath(abspath(@__DIR__), "data", "edge_↑.csv"))
    init_cas_edge_data(joinpath(abspath(@__DIR__), "data", "edge_⤉.csv"))
end

function set_route(M, C, init, goal, w)
    set_init!(M, init, w)
    set_goals!(M, [goal], w)
    generate_transitions!(M, M.graph)
    reset_problem!(M, C)
end

function random_route(M, C)
    init = rand([12, 1, 4, 16])
    goals = rand(5:8, 1)
    while init in goals
        init = rand(1:16)
    end
    set_init!(M, init)
    set_goals!(M, goal)
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
