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
struct COCASstate
       sh::Vector{Int}
    state::DomainState
        σ::Char
end
function ==(a::COCASstate, b::COCASstate)
    return isequal(a.sh, b.sh) && isequal(a.state, b.state) && isequal(a.σ, b.σ)
end
function Base.hash(a::COCASstate, h::UInt)
    h = hash(a.sh, h)
    h = hash(a.state, h)
    h = hash(a.σ, h)
    return h
end

struct COCASaction
    action::DomainAction
         l::Int
end
function ==(a::COCASaction, b::COCASaction)
    return isequal(a.action, b.action) && isequal(a.l, b.l)
end
function Base.hash(a::COCASaction, h::UInt)
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

            # FIXED COMPETENCE
            # κ[s][a] = competence(state, action)

            # LEARNED COMPETENCE
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

            ## LEARNED COMPETENCE REVISED
            if typeof(state) == NodeState && action.value == '⤉'
                κ[s][a] = 2
            elseif typeof(state) == EdgeState && action.value == '↓'
                κ[s][a] = 2
            else
                κ[s][a] = 1
            end
        end
    end
    return κ
end

function update_potential(C, ℒ, s, a, L)
    state = COCASstate([1, 1, 1], C.𝒮.D.S[s], '⊕')
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
        if state.w.time != C.s₀.state.w.time || state.w.weather != C.s₀.state.w.weather
            continue
        end
        for (a, action) in enumerate(C.𝒮.D.A)
            if κ[s][a] in [0, 2]
                continue
            end
            # if typeof(state) == EdgeState
            #     continue
            # end

            L = [0,1,2]
            update_potential(C, ℒ, s, a, L)
            distr = softmax([C.potential[s][a][l+1] for l in L])
            i = sample(aweights(distr))

            if L[i] == 2
                if C.𝒮.F.λ[1][1][s][a][1]['⊕'] < 0.95 || competence(state, action) == 0
                    C.potential[s][a][L[i] + 1] = 0.0
                    continue
                end
            elseif L[i] == 0
                if C.𝒮.F.λ[1][1][s][a][1]['⊕'] > 0.05 || competence(state, action) == 2
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
            if state.r == "None" && !state.o
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

function autonomy_cost(state::COCASstate)
    if state.σ == '⊕'
        return 0.0
    elseif state.σ == '∅'
        if state.sh[3] == 1
            if state.state.w.active_avs == 1
                return 4 * 4.0
            else
                return 4*(state.state.w.active_avs-1.0)
            end #1.0
        else
            return 4.0
        end
    else
        return 2.0
    end
end
##

##
mutable struct OperatorModel
   SH::Set{Vector{Int}} # Operator state vector length n+1
   TH::Function
    Σ::Vector{Char}
    λ::Dict{Any, Any}
    ρ::Function
    D::Dict{Int, Dict{String, Dict{String, Dict{Int, DataFrame}}}}
    ϵ::Float64
end

function generate_random_operator_state()
    o1, o2 = sample(1:2), sample(1:2)
    oa = (o1 == 1) ? 1 : 2
    return [o1, o2, oa]
end

function human_state_transition(sh, s, a, l)
    o1, o2, oa = sh[1], sh[2], sh[3]
    active_avs = s.w.active_avs + 1
    if active_avs == 5
        active_avs = 1
    end
    T = Vector{Tuple{Vector, Float32}}()
    if o1 == 1 # Local operator available --> state is [1, x, 1]
        # Local operator becomes busy (only happens if not using operator)
        if l == 0
            p_becomes_busy = (1.0 - (0.5)^active_avs) / 5
        elseif l == 1
            p_becomes_busy = (1.0 - (0.5)^active_avs) / 2
        else
            p_becomes_busy = (1.0 - (0.5)^active_avs)
        end

        if o2 == 1
            push!(T, ([2, 1, 2], p_becomes_busy * 0.75))
            push!(T, ([2, 2, 2], p_becomes_busy * 0.25))
            push!(T, ([1, 1, 1], (1.0-p_becomes_busy) * 0.75))
            push!(T, ([1, 2, 1], (1.0-p_becomes_busy) * 0.25))
        else
            push!(T, ([2, 1, 2], p_becomes_busy * 0.25))
            push!(T, ([2, 2, 2], p_becomes_busy * 0.75))
            push!(T, ([1, 1, 1], (1.0-p_becomes_busy) * 0.25))
            push!(T, ([1, 2, 1], (1.0-p_becomes_busy) * 0.75))
        end

    else # Local operator unavailable --> state is [2, x, 2]
        p_becomes_active = (0.5)^active_avs
        if o2 == 1
            push!(T, ([1, 1, 1], p_becomes_active * 0.75))
            push!(T, ([1, 2, 1], p_becomes_active * 0.25))
            push!(T, ([2, 1, 2], (1.0 - p_becomes_active) * 0.75))
            push!(T, ([2, 2, 2], (1.0 - p_becomes_active) * 0.25))
        else
            push!(T, ([1, 1, 1], p_becomes_active * 0.25))
            push!(T, ([1, 2, 1], p_becomes_active * 0.75))
            push!(T, ([2, 1, 2], (1.0 - p_becomes_active) * 0.25))
            push!(T, ([2, 2, 2], (1.0 - p_becomes_active) * 0.75))
        end
    end
    return T
end

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

function get_consistency(sh)
    o1, o2, oa = sh[1],sh[2],sh[3]
    if oa == 1
        return 1.0
    else
        if o2 == 1
            return 0.9
        else
            return 0.8
        end
    end
end

function set_consistency(F::OperatorModel, ϵ)
    F.ϵ = ϵ
end

function get_state_features(state::DomainState)
    if typeof(state) == NodeState
        return [state.p state.o state.v state.w.active_avs state.w.time state.w.weather]
    else
        return [state.o state.l state.r state.w.active_avs state.w.time state.w.weather]
    end
end

function generate_feedback_profile(𝒟::DomainSSP,
                                   Σ::Vector{Char},
                                   L::Vector{Int},
                                   D::Dict{Int, Dict{String, Dict{String, Dict{Int, DataFrame}}}})
    S, A = 𝒟.S, 𝒟.A
    λ = Dict(o=>Dict(sh=>Dict(s=>Dict(a=>Dict(l=>Dict(σ => 0.5 for σ ∈ Σ)
                                                               for l=0:1)
                                                               for a=1:length(A))
                                                               for s=1:length(S))
                                                               for sh=1:2)
                                                               for o=1:2)

    ##  SET COMPETENCE VALUES FOR FIXED COCAS
    # Threads.@threads for s=1:length(S)
    #     state = S[s]
    #     for a=1:length(A)
    #         action = A[a]
    #         for o=1:2
    #             for sh=1:2
    #                 for l=0:1
    #                     if o == 1
    #                         σ = generate_feedback(COCASstate([1,1,1],state,'∅'), COCASaction(action,l), 1.0)
    #                         if σ == '⊕'
    #                             λ[o][sh][s][a][l]['⊕'] = 1.
    #                             λ[o][sh][s][a][l]['⊖'] = 0.
    #                         elseif σ == '⊖'
    #                             λ[o][sh][s][a][l]['⊕'] = 0.
    #                             λ[o][sh][s][a][l]['⊖'] = 1.
    #                         elseif σ == '∅'
    #                             λ[o][sh][s][a][l]['∅'] = 1.
    #                             λ[o][sh][s][a][l]['⊘'] = 0.
    #                         else
    #                             λ[o][sh][s][a][l]['∅'] = 0.
    #                             λ[o][sh][s][a][l]['⊘'] = 1.
    #                         end
    #                     else
    #                         if sh == 1
    #                             σ = generate_feedback(COCASstate([2,1,2],state,'∅'), COCASaction(action,l), 1.0)
    #                             if σ == '⊕'
    #                                 λ[o][sh][s][a][l]['⊕'] = .8
    #                                 λ[o][sh][s][a][l]['⊖'] = .2
    #                             elseif σ == '⊖'
    #                                 λ[o][sh][s][a][l]['⊕'] = .2
    #                                 λ[o][sh][s][a][l]['⊖'] = .8
    #                             elseif σ == '∅'
    #                                 λ[o][sh][s][a][l]['∅'] = .8
    #                                 λ[o][sh][s][a][l]['⊘'] = .2
    #                             else
    #                                 λ[o][sh][s][a][l]['∅'] = 0.
    #                                 λ[o][sh][s][a][l]['⊘'] = 1.
    #                             end
    #                         else
    #                             σ = generate_feedback(COCASstate([2,2,2],state,'∅'), COCASaction(action,l), 1.0)
    #                             if σ == '⊕'
    #                                 λ[o][sh][s][a][l]['⊕'] = .7
    #                                 λ[o][sh][s][a][l]['⊖'] = .3
    #                             elseif σ == '⊖'
    #                                 λ[o][sh][s][a][l]['⊕'] = .3
    #                                 λ[o][sh][s][a][l]['⊖'] = .7
    #                             elseif σ == '∅'
    #                                 λ[o][sh][s][a][l]['∅'] = .7
    #                                 λ[o][sh][s][a][l]['⊘'] = .3
    #                             else
    #                                 λ[o][sh][s][a][l]['∅'] = 0.
    #                                 λ[o][sh][s][a][l]['⊘'] = 1.
    #                             end
    #                         end
    #                     end
    #                 end
    #             end
    #         end
    #

    ##
    ## OLD CODE DEPRECATED
    ##
    # for o=1:2
    #     for (a, action) in enumerate(A)
    #         # X_n, Y_n = split_data(D[o]["node"][string(action.value)])
    #         # M_n = build_forest(Y_n, X_n, -1, 10, 0.5, -1)
    #         # if action.value ∈ ['↑', '⤉']
    #         #     X_e, Y_e = split_data(D[o]["edge"][string(action.value)])
    #         #     M_e = build_forest(Y_e, X_e, -1, 10, 0.5, -1)
    #         # end
    #         for (s, state) in enumerate(S)
    #             if typeof(state) == EdgeState && action.value ∉ ['↑', '⤉']
    #                 continue
    #             end
    #             f = get_state_features(state)
    #             for sh=1:2
    #                 for l=0:1
    #                     # if typeof(state) == NodeState
    #                     #     pred = apply_forest_proba(M_n, hcat(f,sh,l), [0,1])
    #                     # else
    #                     #     pred = apply_forest_proba(M_e, hcat(f,sh,l), [0,1])
    #                     # end
    #                     for σ in Σ
    #                         if σ == '⊖' || σ == '⊘'
    #                             λ[o][sh][s][a][l][σ] = 0.5 #pred[1]
    #                         else
    #                             λ[o][sh][s][a][l][σ] = 0.5 #pred[2]
    #                         end
    #                     end
    #                 end
    #             end
    #         end
    #     end
    # end
    return λ
end

function update_feedback_profile!(C)
    λ, 𝒟, Σ, L, D = C.𝒮.F.λ, C.𝒮.D, C.𝒮.F.Σ, C.𝒮.A.L, C.𝒮.F.D
    S, A = 𝒟.S, 𝒟.A
    for o=1:2
        for (a, action) in enumerate(A)
            for l=0:1
                X_n, Y_n, M_n = missing, missing, missing
                failed_to_build_node, failed_to_build_edge = false, false
                try
                    X_n, Y_n = split_data(D[o]["node"][string(action.value)][l])
                    M_n = build_forest(Y_n, X_n, -1, 11, 0.7, -1)
                catch
                    failed_to_build_node = true
                end

                X_e, Y_e, M_e = missing, missing, missing
                if action.value ∈ ['↑', '⤉']
                    try
                        X_e, Y_e = split_data(D[o]["edge"][string(action.value)][l])
                        M_e = build_forest(Y_e, X_e, -1, 11, 0.7, -1)
                    catch
                        failed_to_build_edge = true
                    end
                end

                for (s, state) in enumerate(S)
                    if typeof(state) == EdgeState && action.value ∉ ['↑', '⤉']
                        continue
                    end
                    f = get_state_features(state)
                    for sh=1:2
                        if typeof(state) == NodeState
                            if failed_to_build_node
                                for σ ∈ Σ
                                    λ[o][sh][s][a][l][σ] = 0.5
                                end
                                continue
                            else
                                pred = []
                                try
                                    pred = apply_forest_proba(M_n, hcat(f,sh), [0,1])
                                catch
                                    pred = [0.5 0.5]
                                end
                                if pred[1] == 1.0
                                    pred[1] = 0.99
                                    pred[2] = 0.01
                                elseif pred[2] == 1.0
                                    pred[2] = 0.99
                                    pred[1] = 0.01
                                end
                                for σ in Σ
                                    if σ == '⊖' || σ == '⊘'
                                        λ[o][sh][s][a][l][σ] = pred[1]
                                    else
                                        λ[o][sh][s][a][l][σ] = pred[2]
                                    end
                                end
                            end
                        else
                            if failed_to_build_edge
                                for σ ∈ Σ
                                    λ[o][sh][s][a][l][σ] = 0.5
                                end
                                continue
                            else
                                pred = []
                                try
                                    pred = apply_forest_proba(M_e, hcat(f,sh), [0,1])
                                catch
                                    pred = [0.5 0.5]
                                end
                                if pred[1] == 1.0
                                    pred[1] = 0.99
                                    pred[2] = 0.01
                                elseif pred[2] == 1.0
                                    pred[2] = 0.99
                                    pred[1] = 0.01
                                end
                                for σ in Σ
                                    if σ == '⊖' || σ == '⊘'
                                        λ[o][sh][s][a][l][σ] = pred[1]
                                    else
                                        λ[o][sh][s][a][l][σ] = pred[2]
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return λ
end
#
function save_feedback_profile(λ)
    save_object(joinpath(abspath(@__DIR__),"params.jld2"), λ)
end
#
function load_feedback_profile()
    return load_object(joinpath(abspath(@__DIR__),"params.jld2"))
end
#
function save_data(D)
    for o=1:2
        for k in keys(D[o]["edge"])
            for l=0:1
                record_data(D[o]["edge"][k][l], joinpath(abspath(@__DIR__), "data", "operator_$o", "level_$l", "edge_$k.csv"), false)
            end
        end
        for k in keys(D[o]["node"])
            for l=0:1
                record_data(D[o]["node"][k][l], joinpath(abspath(@__DIR__), "data", "operator_$o", "level_$l", "node_$k.csv"), false)
            end
        end
    end
end

function human_cost(action::COCASaction)
    return [1.0 1.0 0.0][action.l + 1]
end
##

struct COCAS
    D::DomainSSP
    A::AutonomyModel
    F::OperatorModel
end

mutable struct COCASSP
    𝒮::COCAS
    S::Vector{COCASstate}
    A::Vector{COCASaction}
    T
    C::Array{Array{Float64,1},1}
   s₀::COCASstate
    G::Set{COCASstate}
    SIndex::Dict{COCASstate, Int}
    AIndex::Dict{COCASaction, Int}
    potential::Dict{Int, Dict{Int, Vector{Float64}}}
end
function COCASSP(𝒮::COCAS,
               S::Vector{COCASstate},
               A::Vector{COCASaction},
               T::Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}},
               C::Array{Array{Float64,1},1},
              s₀::COCASstate,
               G::Set{COCASstate})
    SIndex, AIndex = generate_index_dicts(S, A)
    s_length, a_length = size(𝒮.D.S)[1], size(𝒮.D.A)[1]
    potential = Dict(s => Dict(a => [0. for i=1:3] for a=1:a_length) for s=1:s_length)
    return COCASSP(𝒮, S, A, T, C, s₀, G, SIndex, AIndex, potential)
end

function generate_index_dicts(S::Vector{COCASstate}, A::Vector{COCASaction})
    SIndex = Dict{COCASstate, Integer}()
    for (s, state) ∈ enumerate(S)
        SIndex[state] = s
    end
    AIndex = Dict{COCASaction, Integer}()
    for (a, action) ∈ enumerate(A)
        AIndex[action] = a
    end
    return SIndex, AIndex
end

function generate_states(D, F)
    states = Vector{COCASstate}()
    G = Set{COCASstate}()
    for sh in F.SH
        for state in D.S
            for σ in F.Σ
                new_state = COCASstate(sh, state, σ)
                push!(states, new_state)
                if state in D.G && σ == '⊕'
                    push!(G, new_state)
                end
            end
        end
    end
    o1, o2 = rand(1:2), rand(1:2)
    oa = (o1 == 1) ? 1 : 2
    sh = [o1, o2, oa]
    return states, COCASstate(sh, D.s₀, '⊕'), G
end

function reset_problem!(D, C)
    sh = generate_random_operator_state()
    C.s₀ = COCASstate(sh, D.s₀, '⊕')
    C.G = Set{COCASstate}()
    for state in D.G
        for sh in C.𝒮.F.SH
            push!(C.G, COCASstate(sh, state, '⊕'))
        end
    end
    generate_costs!(D)
    generate_costs!(C)
end

function terminal(C::COCASSP, state::COCASstate)
    return state in C.G
end

function generate_actions(D, A)
    actions = Vector{COCASaction}()
    for action in D.A
        for l in A.L
            new_action = COCASaction(action, l)
            push!(actions, new_action)
        end
    end
    return actions
end

function allowed(C, s::Int, a::Int)
    return C.A[a].l <= C.𝒮.A.κ[C.𝒮.D.SIndex[C.S[s].state]][Int(ceil(a/3))]
end

function generate_transitions!(𝒟, 𝒜, ℱ, C,
                              S::Vector{COCASstate},
                              A::Vector{COCASaction},
                              G::Set{COCASstate})

    T = C.T
    κ, λ = 𝒜.κ, ℱ.λ
    for s = 1:length(S)#(s, state) in enumerate(S)
        state = S[s]
        if state.state.w.time != C.s₀.state.w.time || state.state.w.weather != C.s₀.state.w.weather
            continue
        end
        if state.sh == [1, 1, 2]
            continue
        end
        T[s] = Dict{Int, Vector{Tuple{Int, Float64}}}()
        for (a, action) in enumerate(A)
            if state.state in 𝒟.G
                state′ = COCASstate(state.sh, state.state, '⊕')
                T[s][a] = [(C.SIndex[state′], 1.0)]
                continue
            end

            if (typeof(state.state) == EdgeState && !state.state.o && action.action.value == '⤉')
                T[s][a] = [(s, 1.0)]
                continue
            end

            base_state = state.state
            base_action = action.action
            base_s = 𝒟.SIndex[base_state]
            base_a = 𝒟.AIndex[base_action]

            th = ℱ.TH(state.sh, base_state, base_action, action.l)
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
                for i=1:length(th)
                    if action.l == 2
                        push!(T[s][a], (C.SIndex[COCASstate(th[i][1],
                                        dstate′, '⊕')], th[i][2]))
                    else
                        push!(T[s][a], (C.SIndex[COCASstate(th[i][1],
                                        dstate′, state.σ)], th[i][2]))
                    end
                end
                continue
            end

            T[s][a] = Vector{Tuple{Int, Float64}}()
            if action.l == 0
                p_approval = λ[state.sh[3]][state.sh[state.sh[3]]][base_s][base_a][0]['∅']
                p_disapproval = λ[state.sh[3]][state.sh[state.sh[3]]][base_s][base_a][0]['⊘']

                if typeof(state.state) == EdgeState
                    for i=1:length(th)
                        if state.state.o && action.action.value == '⤉'
                            dstate′ = EdgeState(state.state.u,
                                    state.state.v, state.state.θ, false,
                                    state.state.l, state.state.r, w)
                            state′ = COCASstate(th[i][1], dstate′, '∅')
                            push!(T[s][a], (C.SIndex[state′], th[i][2] * p_approval))
                            dstate′′ = EdgeState(state.state.u, state.state.v,
                                    state.state.θ, state.state.o, state.state.l,
                                    state.state.r, w)
                            push!(T[s][a], (C.SIndex[COCASstate(th[i][1], dstate′′, '⊘')], th[i][2] * p_disapproval))
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
                                state′ = COCASstate(th[i][1], 𝒟.S[temp[j][1]], '∅')
                                push!(T[s][a], (C.SIndex[state′], (temp[j][2]/mass)*p_approval*th[i][2]))
                            end
                            dstate′ = EdgeState(state.state.u, state.state.v,
                                    state.state.θ, state.state.o, state.state.l,
                                    state.state.r, w)
                            push!(T[s][a], (C.SIndex[COCASstate(th[i][1], dstate′, '⊘')], th[i][2]*p_disapproval))
                        else
                            dstate′ = EdgeState(state.state.u, state.state.v,
                                    state.state.θ, state.state.o, state.state.l,
                                    state.state.r, w)
                            push!(T[s][a], (C.SIndex[COCASstate(th[i][1], dstate′, '∅')], th[i][2]))
                        end
                    end
                else
                    for i = 1:length(th)
                        for j = 1:length(t)
                            push!(T[s][a], (C.SIndex[COCASstate(th[i][1],
                              𝒟.S[t[j][1]], '∅')], th[i][2] * t[j][2] * p_approval))
                        end
                        dstate′ = NodeState(state.state.id, state.state.p,
                            state.state.o, state.state.v, state.state.θ, w)
                        push!(T[s][a], (C.SIndex[COCASstate(th[i][1], dstate′, '⊘')],
                                        th[i][2] * p_disapproval))
                    end
                end
            elseif action.l == 1
                p_approve = λ[state.sh[3]][state.sh[state.sh[3]]][base_s][base_a][1]['⊕']
                p_disapprove = 1.0 - p_approve
                if typeof(state.state) == NodeState
                    dstate′ = NodeState(state.state.id, state.state.p,
                        state.state.o, state.state.v, state.state.θ, w)
                else
                    dstate′ = EdgeState(state.state.u, state.state.v,
                        state.state.θ, state.state.o, state.state.l, state.state.r, w)
                end
                for i=1:length(th)
                    push!(T[s][a], (C.SIndex[COCASstate(th[i][1], dstate′, '⊖')],
                                    th[i][2] * p_disapprove))
                    for j=1:length(t)
                        push!(T[s][a], (C.SIndex[COCASstate(th[i][1], 𝒟.S[t[j][1]], '⊕')],
                                th[i][2] * t[j][2] * p_approve))
                    end
                end
            else
                for i=1:length(th)
                    for j=1:length(t)
                        push!(T[s][a], (C.SIndex[COCASstate(th[i][1],
                                𝒟.S[t[j][1]], '⊕')], th[i][2] * t[j][2]))
                        # push!(T[s][a], ((sp-1) * 4 + 4, p))
                    end
                end
            end
        end
    end
end

function check_transition_validity(C)
    S, A, T = C.S, C.A, C.T
    for (s, state) in enumerate(S)
        if state.state.w != C.s₀.state.w || state.sh == [1, 1, 2]
            continue
        end
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
                break
            end
        end
    end
end

function block_transition!(C::COCASSP,
                       state::COCASstate,
                      action::COCASaction)
    state′ = COCASstate(state.sh, state.state, '⊕')
    s, a = C.SIndex[state′], C.AIndex[action]
    # TODO: why do we not block C.T[s][a] as well? Not understanding..
    for i=0:15
        if s+i in keys(C.T)
            C.T[s+i][a] = [(s+i, 1.0)]
        end
    end
    # C.T[s][a] = [(s, 1.0)]
    # C.T[s+1][a] = [(s+1, 1.0)]
    # C.T[s+2][a] = [(s+2, 1.0)]
    # C.T[s+3][a] = [(s+3, 1.0)]
end

## TODO: BIG PROBLEM WE ARE ADDING IN THE DOMAIN COST EVEN WHEN THE ACTION
#        IS DENIED OR THE REQUEST FOR TRANSFER OF CONTROL IS DENIED.

function generate_costs(C::COCASSP,
                        s::Int,
                        a::Int,)
    D, A, F = C.𝒮.D, C.𝒮.A, C.𝒮.F
    state, action = C.S[s], C.A[a]
    cost = D.C[D.SIndex[state.state]][D.AIndex[action.action]]
    cost += A.μ(state)
    cost += F.ρ(action)
    return cost
end

function generate_costs!(C::COCASSP)
    Threads.@threads for s = 1:length(C.S)
        for a = 1:length(C.A)
            C.C[s][a] = generate_costs(C, s, a)
        end
    end
    # C.C = [[generate_costs(C, s, a) for a=1:length(C.A)] for s=1:length(C.S)]
end

function generate_feedback(state::COCASstate,
                          action::COCASaction,
                               ϵ::Float64)
    sh = state.sh
    w = state.state.w
    # Request for ToC logic
    if action.l == 0
        # Operator noise            ----> No operator noise in ToC
        # if rand() < 1 - ϵ
        #     return ['⊘', '∅'][rand(1:2)]
        # end

        if sh[3] == 1 # Local operator always accepts
            return '∅'
        else
            if sh[2] == 1
                # Global operator accepts if:
                #   - No pedestrians
                #   - No occlusions
                #   - < 4 vehicle complexity
                #   - daytime and sunny
                if w.time == "day" && w.weather == "sunny"
                    if ((typeof(state.state) == EdgeState && state.state.l > 1) ||
                        (typeof(state.state) == NodeState && !state.state.p &&
                          !state.state.o && state.state.v < 3))
                        if rand() < 1 - ϵ
                            return '⊘'
                        else
                            return '∅'
                        end
                    end
                    return '⊘'
                elseif ((w.time == "day" && w.weather == "rainy") ||
                        (w.time == "night" && w.weather == "sunny"))
                    if ((typeof(state.state) == EdgeState && state.state.l > 2) ||
                        (typeof(state.state) == NodeState && !state.state.p &&
                         !state.state.o && state.state.v < 1))
                        if rand() < 1 - ϵ
                            return '⊘'
                        else
                            return '∅'
                        end
                    end
                    return '⊘'
                else
                    return '⊘'
                end
            else # Global operator always rejects with bad connection
                return '⊘'
            end
        end
    end

    if typeof(state.state) == EdgeState && action.action.value == '↑'
        if state.state.o
            return '⊖'
        else
            if state.state.r == "None"
                if rand() < 1 - ϵ
                    return '⊖'
                else
                    return '⊕'
                end
            else
                return '⊖'
            end
        end
    end

    if sh[3] == 2
        if state.sh[2] == 1
            if (state.state.w.time == "night" && state.state.w.weather == "snowy")
                return '⊖'
            end
        else
            if (state.state.w.time == "night" && state.state.w.weather == "rainy" ||
                state.state.w.weather == "snowy")
                return '⊖'
            end
        end
    end

    if rand() < 1 - ϵ
        return ['⊕', '⊖'][rand(1:2)]
    end

    if typeof(state.state) == EdgeState
        if action.action.value == '⤉'
            if state.state.o && state.state.l == 1
                return '⊖'
            else
                return '⊕'
            end
        else
            return '⊕'
        end
    else
        if action.action.value == '⤉'
            return '⊕'
        elseif action.action.value == '→'
            if ((state.state.o || state.state.w.weather == "snowy" ||
               (state.state.w.time == "night" && state.state.w.weather == "rainy"))
                && state.state.p && state.state.v > 1)
                return '⊖'
            else
                return '⊕'
            end
        else
            if (state.state.o || state.state.w.weather == "snowy" ||
               (state.state.w.time == "night" && state.state.w.weather == "rainy"))
                if state.state.p || state.state.v > 1
                    return '⊖'
                else
                    return '⊕'
                end
            else
                if state.state.p && state.state.v > 2
                    return '⊖'
                else
                    return '⊕'
                end
            end
        end
    end
end

function generate_successor(M::DomainSSP,
                        state::COCASstate,
                       action::COCASaction,
                            σ::Char)
    s, a = M.SIndex[state.state], M.AIndex[action.action]
    thresh = rand()
    p = 0.
    T = M.T[s][a]
    for (s′, prob) ∈ T
        p += prob
        if p >= thresh
            TH = human_state_transition(state.sh, state.state, action.action, action.l)
            sh = sample(first.(TH), aweights(last.(TH)))
            return COCASstate(sh, M.S[s′], σ)
        end
    end
end

function generate_successor(M::COCASSP,
                            s::Int,
                            a::Int,
                            σ::Char)
    state, action = M.S[s], M.A[a]
    TH = human_state_transition(state.sh, state.state, action.action, action.l)
    sh = sample(first.(TH), aweights(last.(TH)))

    state = M.S[sample(first.(M.T[s][a]), aweights(last.(M.T[s][a])))]
    if terminal(M, state)
        return state
    end
    i = 1
    while state.sh != sh || state.σ != σ
        # println(sample(first.(M.T[s][a]), aweights(last.(M.T[s][a]))))
        # println(s, " | ", a, " | ", σ)
        state = M.S[sample(first.(M.T[s][a]), aweights(last.(M.T[s][a])))]
        i += 1
        if i > 1000
            println("ERROR")
            println(s, " | ", a, " | ", σ)
            @assert false
        end
    end

    return state
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

function compute_reachable_level_optimality(C, L)
    κ = C.𝒮.A.κ
    R = reachable(C, L)
    total_full = 0
    level_optimal_full = 0
    total_policy = 0
    level_optimal_policy = 0
    for s in R
        if terminal(C, C.S[s])
            continue
        end
        # Level optimal on policy only
        state = C.S[s]
        ds = C.𝒮.D.SIndex[state.state]
        a = L.π[s]
        action = C.A[a]
        total_policy += 1
        level_optimal_policy += (κ[ds][C.𝒮.D.AIndex[action.action]] == competence(state.state, action.action))

        # Level optimal on all
        for da in keys(κ[ds])
            total_full += 1
            level_optimal_full += κ[ds][da] == competence(C.𝒮.D.S[ds], C.𝒮.D.A[da])
        end
    end

    return (level_optimal_full/total_full), (level_optimal_policy/total_policy)
end

function compute_level_optimality(C, ℒ)
    println("Computing level optimality...")
    total = 0
    r = 0
    lo = 0
    lo_r = 0
    W = vec(collect(Base.product(
        1:4, ["day", "night"], ["sunny", "rainy", "snowy"]
    )))
    # R = reachable(C, ℒ)
    for w in W
        println("Computing for world state $w")
        set_route(C.𝒮.D, C, C.s₀.state.id, pop!(C.G).state.id, WorldState(w))
        generate_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)
        for (s, state) in enumerate(C.S)
            if (terminal(C, state) || state.state.w.time != C.s₀.state.w.time ||
                state.state.w.weather != C.s₀.state.w.weather)
                continue
            end
            if s ∉ keys(C.T)
                continue
            end
            solve(ℒ, C, s)
            total += 1
            action = C.A[ℒ.π[s]]
            comp = (action.l == competence(state.state, action.action))
            lo += comp
            if s in R
                r += 1
                lo_r += comp
            end
        end
    end
    return 100.0 * lo/total, 100.0 * lo_r/r
end

function compute_level_optimality(C, visited)
    κ = C.𝒮.A.κ
    D = C.𝒮.D
    total = 0
    level_optimal = 0
    total_visited = 0
    level_optimal_visited = 0

    for s in keys(κ)
        for a in keys(κ[s])
            if typeof(D.S[s]) == EdgeState && D.A[a].value ∈ ['←', '→']
                continue
            end
            if terminal(C, C.S[s])
                continue
            end
            total += 1
            level_optimal += (κ[s][a] == competence(D.S[s], D.A[a]))
            if s in visited
                total_visited += 1
                level_optimal_visited += (κ[s][a] == competence(D.S[s], D.A[a]))
            end
        end
    end

    return float(level_optimal / total), float(level_optimal_visited / total_visited)
end

function build_cocas(𝒟::DomainSSP,
                   L::Vector{Int},
                   Σ::Vector{Char})
    D = Dict{Int, Dict{String, Dict{String, Dict{Int, DataFrame}}}}()
    for o=1:2
        D[o] = Dict{String, Dict{String, Dict{Int, DataFrame}}}()
        for t in ["node", "edge"]
            D[o][t] = Dict{String, Dict{Int, DataFrame}}()
            for a in ["↑", "→", "↓", "←", "⤉"]
                D[o][t][a] = Dict{Int, DataFrame}()
                for l=0:1
                    D[o][t][a][l] = (DataFrame(CSV.File(joinpath(abspath(@__DIR__),
                            "data", "operator_$o", "level_$l", "$(t)_$a.csv"))))
                end
            end
        end
    end

    if isdir(joinpath(abspath(@__DIR__), "COCAS_params.jld2"))
        κ, λ = load_object(joinpath(abspath(@__DIR__), "COCAS_params.jld2"))
    else
        κ = generate_autonomy_profile(𝒟)
        λ = generate_feedback_profile(𝒟, Σ, L, D)
    end

    SH = Set([i for i in x] for x in vec(collect(Base.product(1:2, 1:2, 1:2))))
    𝒜 = AutonomyModel(L, κ, autonomy_cost)
    ℱ = OperatorModel(SH, human_state_transition, Σ, λ, human_cost, D, 0.9)
    𝒮 = COCAS(𝒟, 𝒜, ℱ)
    S, s₀, G = generate_states(𝒟, ℱ)
    A = generate_actions(𝒟, 𝒜)
    T = Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}}()
    costs = [[0. for a=1:length(A)] for s=1:length(S)]
    C = COCASSP(𝒮, S, A, T, costs, s₀, G)
    generate_costs!(C)
    generate_transitions!(𝒟, 𝒜, ℱ, C, S, A, G)
    check_transition_validity(C)
    return C
end

function solve_model(C::COCASSP)
    L = solve_model(C.𝒮.D)
    # H = [L.V[M.SIndex[state.state]] for (s,state) in enumerate(C.S)]
    # ℒ = LAOStarSolver(100000, 1000., 1.0, .001, Dict{Integer, Integer}(),
    #                     zeros(length(C.S)), zeros(length(C.S)),
    #                     H, zeros(length(C.A)),
    #                     [false for i=1:length(C.S)])
    ℒ = LRTDPsolver(C, 1000., 1000, .01, Dict{Int, Int}(),
                     false, Set{Int}(), L.V, zeros(length(C.S)),
                                             zeros(length(C.A)))
    solve(ℒ, C, C.SIndex[C.s₀])
    return ℒ
end

function init_data()
    for o=1:2
        for action in ["←", "↑", "→", "↓", "⤉"]
            for l=0:1
                init_cocas_node_data(joinpath(abspath(@__DIR__), "data",
                "operator_$o", "level_$l", "node_$action.csv"))
                init_cocas_edge_data(joinpath(abspath(@__DIR__), "data",
                "operator_$o", "level_$l", "edge_$action.csv"))
            end
        end
    end
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

function debug_competence(C)
    κ, λ, D = C.𝒮.A.κ, C.𝒮.F.λ, C.𝒮.D
    # for (s, state) in enumerate(C.S)
    for s in keys(κ)
        # println("**** $s ****")
        state = D.S[s]
        for a in keys(κ[s])
            if typeof(D.S[s]) == EdgeState && D.A[a].value ∈ ['←', '→']
                continue
            end
            action = D.A[a]
            if κ[s][a] != competence(D.S[s], D.A[a])
                println("-----------------------")
                println("State:  $state      $s |       Action: $action         $a")
                println("Competence: $(competence(state, action))")
                println("Kappa: $(κ[s][a])")
                println("Lambda: $(λ[1][1][s][a])")
                println("-----------------------")
            end
        end
        # ds = Int(ceil(s/4))
        # a = solve(L, C, s)[1]
        # action = C.A[a]
        # da = Int(ceil(a/4))
        # if action.l != competence(state.state, action.action)
        #     println("-----------------------")
        #     println("State:  $state      $s |       Action: $action         $a")
        #     println("Competence: $(competence(state.state, action.action))")
        #     println("Kappa: $(κ[ds][da])")
        #     println("Lambda: $(λ[ds][da])")
        #     println("-----------------------")
        # end
    end
end
# debug_competence(C, L)
