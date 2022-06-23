import Combinatorics

using Plots
# using GLM
using DecisionTree
using DataFrames
using CSV
using JLD
using StatsBase

include("domain_model.jl")
# include("../LAOStarSolver.jl")
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

function generate_autonomy_profile(𝒟::DomainSSP,
                                   L::Vector{Int})
    κ = Dict{Int, Dict{Int, Int}}()
    for (s, state) in enumerate(𝒟.S)
        κ[s] = Dict{Int, Int}()
        for (a, action) in enumerate(𝒟.A)
            κ[s][a] = 1
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
        for (a, action) in enumerate(C.𝒮.D.A)
            if κ[s][a] == competence(state, action)
                continue
            end

            # L = [κ[s][a]-1, κ[s][a], κ[s][a]+1]
            L = [0, 1, 2]
            update_potential(C, ℒ, s, a, L)
            distr = softmax([C.potential[s][a][l+1] for l in L])
            i = sample(aweights(distr))

            if L[i] == 2
                if C.𝒮.F.λ[s][a][1]['∅'] < 0.85
                    C.potential[s][a][L[i]+1] = 0.0
                    continue
                end
            elseif L[i] == 0
                if C.𝒮.F.λ[s][a][1]['∅'] > 0.25
                    C.potential[s][a][L[i]+1] = 0.0
                    continue
                end
            elseif L[i] == κ[s][a]
                C.potential[s][a][L[i]+1] = 0.0
                continue
            end

            if L[i] == competence(state, action)
                println("Updated to competence: ($s, $a) | $(κ[s][a]) | $(L[i])")
            end

            C.𝒮.A.κ[s][a] = L[i]
            C.potential[s][a][L[i]+1] = 0.0
        end
    end
end

function competence(state::DomainState,
                   action::DomainAction)
    if state.position == 4
        return 2
    end
    if state.position == -1
        return -1
    end
    if action.value == :stop
        if state.position > 1 || (state.oncoming < 1 && state.trailing)
            return 0
        else
            return 2
        end
    elseif action.value == :edge
        if state.position > 0 && state.trailing
            return 0
        else
            return 2
        end
    else
        if state.oncoming == -1
            return 0
        elseif state.oncoming > 0
            return 1
        else
            return 2
        end
    end
end

function save_autonomy_profile(κ)
    save(joinpath(abspath(@__DIR__), "params.jld"), "κ", κ)
end

function load_autonomy_profile()
    return load(joinpath(abspath(@__DIR__), "params.jld"), "κ")
end

function autonomy_cost(state::CASstate)
    if state.σ == '∅'
        return 0.0
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
    D::Dict{String, DataFrame}
    D_full::Dict{String, DataFrame}
end

function get_state_features(state::DomainState)
    x = [state.position state.oncoming state.priority]
    return x
end

function get_full_state_features(state::DomainState)

end

function onehot(x)
    return transpose(vcat(Flux.onehot(x[1], 0:4), Flux.onehot(x[2], -1:3), x[3], x[4]))
end

function generate_feedback_profile(𝒟::DomainSSP,
                                   Σ::Vector{Char},
                                   L::Vector{Int},
                                   D::Dict{String, DataFrame})
   S, A = 𝒟.S, 𝒟.A
   λ = Dict(s => Dict(a => Dict(1 => Dict(σ => 0.5 for σ ∈ Σ))
                                                   for a=1:length(A))
                                                   for s=1:length(S))
    for (a, action) in enumerate(A)
        X, Y = split_data(D[string(action.value)])
        M = build_forest(Y, X, 2, 10, 0.5, 8)
        for (s, state) in enumerate(S)
            if state.position == -1
                λ[s][a] = Dict(1 => Dict('∅' => 1.0, '⊘' => 0.0))
                continue
            end
            f = get_state_features(state)
            pred = apply_forest_proba(M, f, [0,1])
            λ[s][a][1]['⊘'] = pred[1]
            λ[s][a][1]['∅'] = pred[2]
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
            if state.position == -1
                continue
            end
            f = get_state_features(state)
            pred = apply_forest_proba(M, f, [0,1])
            λ[s][a][1]['⊘'] = pred[1]
            λ[s][a][1]['∅'] = pred[2]
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
    return [5.0 0.5 0.0][action.l+1]
end

function find_candidates(C::CAS, δ=0.05, threshold=60)
    λ, 𝒟, Σ, L, D = C.𝒮.F.λ, C.𝒮.D, C.𝒮.F.Σ, C.𝒮.A.L, C.𝒮.F.D
    S, A = 𝒟.S, 𝒟.A

    active_features = []

    candidates = Set()
    for s in keys(λ)
        state = S[s]
        f = get_state_features(state)
        for a in keys(λ[s])
            action = A[a]
            count = nrow(groupby(D[string(action.value)], active_features)[Tuple(f)])

            if count < threshold
                continue
            end

            candidate = True

            for σ ∈ Σ
                if λ[s][a][σ] > 1 - δ
                    candidate = false
                end
            end

            if candidate
                push!(candidates, [state action])
            end
        end
    end

    return candidates
end

function mRMR(df, y)
    relevance = 0.
    X = Matrix(one_hot_encode(DataFrame([df], :auto), drop_original=true))
    relevance = sum(f_test(Matrix(X), y)) / size(X)[2]

    repetition = 0.
    for i = 1:size(X)[2]
        repetition += sum(pearson_correlation(X, X[:, i])) - 1.0
    end
    repetition = repetition / size(X)[2]
    return relevance / repetition
end

function build_lambda(D_train, features, discriminator)
    X = D_train[!, cat(features, discriminator; dims=1)]
    Y = D_train[!, :σ]
    λ = build_forest(Y, X, 2, 10, 0.5, 8)
    return λ
end

function test_lambda(λ, D_test, Y, features, discriminator)
    X = D_test[!, cat(features, discriminator; dims=1)]
    Y = D_test[!, :σ]
    preds = apply_forest_proba(M, X, [0,1]) .> 0.5
    return mcc(Y, preds)
end

function test_discriminators(D_train, D_test, F, discriminators)
    lambdas = [build_lambda(D_train, F, d) for d in discriminators]
    scores = [test_lambda(lambdas[i], D_test, F, discriminators[i] for i=1:length(lambdas)]

    best = argmax(scores)
    best_score = scores[best]
    best_discriminator = discriminators[best]

    curr_score = test_lambda()
    if best_score < curr_score + 0.1 || best_score < 0.5 || curr_score == -1.0
        return None
    else
        return best_discriminator
    end
end

function get_discriminator(C::CAS, candidate, k, scoring_function="mRMR")
    λ, 𝒟, Σ, L = C.𝒮.F.λ, C.𝒮.D, C.𝒮.F.Σ, C.𝒮.A.L
    D, D_full =  C.𝒮.F.D[string(action.value)], C.𝒮.F.D_full(string(action.value))
    S, A = 𝒟.S, 𝒟.A
    state, action = candidate[1], candidate[2]

    inactive_features = []
    if length(inactive_features) == 0
        return -1
    end

    if scoring_function == "mRMR"
        _disc = Dict()
        for f in inactive_features
            _disc[f] = mRMR(D[!, f], D[!, :σ])
        end
    end

    discrims = sort(collect(dict), by = x->x[2])
    D_train, D_test = split_df(D, 0.75)
    return test_discriminators(D_train, D_test, D[]
    D.columns.drop(np.append(unused_features, 'feedback')), discriminators)
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
    # flags::Dict{Int, Dict{Int, Bool}}
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
    # flags = Dict(s => Dict(a => false for a=1:a_length) for s=1:s_length)
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
            if state in D.G && σ == '∅'
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

function allowed(C, s::Int,
                    a::Int)
    return C.A[a].l <= C.𝒮.A.κ[ceil(s/2)][ceil(a/3)]
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
            if state.state in 𝒟.G
                state′ = CASstate(state.state, '∅')
                T[s][a] = [(C.SIndex[state′], 1.0)]
                continue
            end

            if state.state.position == -1
                state′ = CASstate(last(𝒟.S), '∅')
                T[s][a] = [(C.SIndex[state′], 1.0)]
            end

            base_state = state.state
            base_action = action.action
            base_s = 𝒟.SIndex[base_state]
            base_a = 𝒟.AIndex[base_action]

            t = 𝒟.T[base_s][base_a]
            if size(t) == 1 && 𝒟.S[t[1][1]].position == -1
                T[s][a] = [(C.SIndex[CASstate(𝒟.S[t[1][1]], '∅')], 1.0)]
                continue
            end
            if t == [(base_s, 1.0)] || action.l != κ[base_s][base_a]
                T[s][a] = [(s, 1.0)]
                continue
            end

            T[s][a] = Vector{Tuple{Int, Float64}}()
            if action.l == 0
                # T[s][a] = [((t[argmax([x[2] for x in t])][1]-1) * 2 + 2 , 1.0)]
                state′ = CASstate(DomainState(4, 0, 0, state.state.dynamic, 0), '∅')
                push!(T[s][a], (C.SIndex[state′], 1.0))
            elseif action.l == 1
                p_override = ℱ.λ[base_s][base_a][1]['⊘']
                p_null = 1.0 - p_override
                state′ = CASstate(DomainState(4, 0, 0, state.state.dynamic, 0), '⊘')
                push!(T[s][a], (C.SIndex[state′], p_override))
                for (sp, p) in t
                    push!(T[s][a], ((sp-1) * 2 + 2, p * p_null))
                end
            else
                for (sp, p) in t
                    push!(T[s][a], ((sp-1) * 2 + 2, p))
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
    T = C.T
    state′ = CASstate(state.state, '∅')
    s, a = C.SIndex[state′], C.AIndex[action]
    # T[s][a] = [(s, 1.0)]
    T[s+1][a] = [(s+1, 1.0)]
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

function generate_feedback(state::DomainState,
                          action::DomainAction)
    if rand() <= 0.1
        return ['∅', '⊘'][rand(1:2)]
    end
    if state.position == 4
        return '∅'
    end

    if action.value == :stop
        if state.position > 1 || (state.oncoming < 1 && state.trailing)
            return '⊘'
        else
            return '∅'
        end
    elseif action.value == :edge
        if state.position > 0 && state.trailing
            return '⊘'
        else
            return '∅'
        end
    else
        if state.oncoming == -1 || (state.oncoming > 1 && !state.priority)
            return '⊘'
        else
            return '∅'
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
        κ = generate_autonomy_profile(𝒟, L)
    end
    𝒜 = AutonomyModel(L, κ, autonomy_cost)


    D = Dict{String, DataFrame}()
    for a in ["stop", "edge", "go"]
        D[a] = DataFrame(CSV.File(joinpath(abspath(@__DIR__), "data", "$a.csv")))
    end
    λ = generate_feedback_profile(𝒟, Σ, L, D)
    ℱ = FeedbackModel(Σ, λ, human_cost, D)
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
    for action in [:stop, :edge, :go]
        init_pass_obstacle_data(joinpath(abspath(@__DIR__), "data", "$action.csv"))
    end
end

function debug_competence(C, L)
    κ, λ, D = C.𝒮.A.κ, C.𝒮.F.λ, C.𝒮.D
    total, lo = 0,0
    # for s in reachable(C, L)
    #     state = C.S[s]
    #     if terminal(C, state) || state.state.position == -1
    #         continue
    #     end
    for (s, state) in enumerate(C.S)
        println("**** $s ****")
        if terminal(C, state)
            continue
        end
        total += 1
        ds = Int(ceil(s/2))
        a = solve(L, C, s)[1]
        action = C.A[a]
        da = Int(ceil(a/3))
        if action.l != competence(state.state, action.action)
            println("-----------------------")
            println("State:  $state      $s |       Action: $action         $a")
            println("Competence: $(competence(state.state, action.action))")
            println("Kappa: $(κ[ds][da])")
            println("Lambda: $(λ[ds][da])")
            println("-----------------------")
        else
        #     println("-----------------------")
        #     println("State:  $state      $s |       Action: $action         $a")
        #     println("Competence: $(competence(state.state, action.action))")
        #     println("Kappa: $(κ[ds][da])")
        #     println("Lambda: $(λ[ds][da])")
        #     println("-----------------------")
            lo += 1
        end
    end
    println(lo)
    println(total)
    println(lo/total)
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
