import Combinatorics

using Plots
using GLM
using DataFrames
using CSV
using JLD

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
    κ::Dict{Int, Dict{Int, Int}}
    μ
end

function generate_autonomy_profile(𝒟::DomainSSP)
    κ = Dict{Int, Dict{Int, Int}}()
    for (s, state) in enumerate(𝒟.S)
        κ[s] = Dict{Int, Int}()
        for (a, action) in enumerate(𝒟.A)
            if typeof(state) == NodeState
                if action.value == '⤉' || action.value == '↓'
                    κ[s][a] = 3
                elseif state.p == true || state.o == true || state.v > 1
                    κ[s][a] = 1
                else
                    κ[s][a] = 3
                end
            else
                if state.o == true
                    κ[s][a] = 1
                else
                    κ[s][a] = 3
                end
            end
        end
    end
    return κ
end

function update_potential(C, ℒ, s, a, L)
    state = CASstate(C.𝒮.D.S[s], '∅')
    s2 = C.SIndex[state]
    X = [lookahead(ℒ, C, s2, ((a - 1) * 4 + l + 1) ) for l ∈ L]
    P = .75 .* softmax(-1.0 .* X)
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
            if κ[s][a] == 3 || κ[s][a] == 0
                continue
            end
            if typeof(state) == EdgeState && action.value ∉ ['↑', '⤉']
                continue
            end

            L = [κ[s][a]-1, κ[s][a], κ[s][a]+1]
            update_potential(C, ℒ, s, a, L)

            # r = rand()
            # t = 0.0
            for i in sortperm(-[C.potential[s][a][l+1] for l in L])
                # t +=
                if rand() <= C.potential[s][a][L[i]+1]
                    if L[i] == 3
                        if C.𝒮.F.λ[s][a][2]['∅'] < 0.85
                            C.potential[s][a][L[i] + 1] = 0.0
                            break
                        end
                    elseif L[i] == 0
                        if C.𝒮.F.λ[s][a][1]['⊕'] > 0.35
                            C.potential[s][a][L[i] + 1] = 0.0
                            break
                        end
                    elseif L[i] == κ[s][a]
                        C.potential[s][a][L[i] + 1] = 0.0
                        break
                    end
                    if L[i] == competence(state, action)
                        println("Updated to competence: ($s, $a) | $(κ[s][a]) | $(L[i])")
                    end
                    C.𝒮.A.κ[s][a] = L[i]
                    C.potential[s][a][L[i] + 1] = 0.0
                    if L[2] == 1 && L[i] == 2
                        C.flags[s][a] = true
                    end
                    break
                end
            end
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
        if action.value == '⤉' || action.value == '↓'
            return 3
        elseif action == '→'
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
                logit = lm(fm, hcat(X, Y), contrasts= Dict(:x3 => DummyCoding()))
            else
                if action.value ∉ ['↑', '⤉']
                    continue
                end
                X, Y = read_data(joinpath(abspath(@__DIR__), "data", "edge_$(action.value).csv"))
                fm = @formula(y ~ x1 + x2 + x3)
                logit = lm(fm, hcat(X, Y), contrasts= Dict(:x2 => DummyCoding()))
            end
            λ[s][a] = Dict{Int, Dict{Char, Float64}}()
            for l in [1,2]
                λ[s][a][l] = Dict{Char, Float64}()
                for σ ∈ Σ
                    q = DataFrame(hcat(f, l), :auto)
                    p = clamp(predict(logit, q)[1], 0.0, 1.0)
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

function update_feedback_profile!(C)
    λ, 𝒟, Σ, L = C.𝒮.F.λ, C.𝒮.D, C.𝒮.F.Σ, C.𝒮.A.L
    for (s, state) in enumerate(𝒟.S)
        f = get_state_features(state)
        λ[s] = Dict{Int, Dict{Int, Dict{Char, Float64}}}()
        for (a, action) in enumerate(𝒟.A)
            if typeof(state) == NodeState
                X, Y = read_data(joinpath(abspath(@__DIR__), "data", "node_$(action.value).csv"))
                fm = @formula(y ~ x1 + x2 + x3 + x4)
                logit = lm(fm, hcat(X, Y), contrasts= Dict(:x3 => DummyCoding()))
            else
                if action.value ∉ ['↑', '⤉']
                    continue
                end
                X, Y = read_data(joinpath(abspath(@__DIR__), "data", "edge_$(action.value).csv"))
                fm = @formula(y ~ x1 + x2 + x3)
                logit = lm(fm, hcat(X, Y), contrasts= Dict(:x2 => DummyCoding()))
            end

            λ[s][a] = Dict{Int, Dict{Char, Float64}}()
            for l in [1,2]
                λ[s][a][l] = Dict{Char, Float64}()
                for σ ∈ Σ

                    q = DataFrame(hcat(f, l), :auto)
                    p = clamp(predict(logit, q)[1], 0.0, 1.0)

                    if σ == '⊕' || σ == '∅'
                        λ[s][a][l][σ] = p
                    else
                        λ[s][a][l][σ] = 1.0 - p
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

function human_cost(action::CASaction)
    return [8.0 3. 1. 0.][action.l + 1]              #TODO: Fix this.
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
    T = C.T
    state′ = CASstate(state.state, '⊕')
    s, a = C.SIndex[state], C.AIndex[action]
    T[s][a] = [(s, 1.0)]
    T[s+1][a] = [(s+1, 1.0)]
    T[s+2][a] = [(s+2, 1.0)]
    T[s+3][a] = [(s+3, 1.0)]
end

function generate_costs(C::CASSP,
                        s::Int,
                        a::Int,)
    D, A, F = C.𝒮.D, C.𝒮.A, C.𝒮.F
    state, action = C.S[s], C.A[a]
    cost = D.C(D, D.SIndex[state.state], D.AIndex[action.action])
    cost += A.μ(state)
    cost += F.ρ(action)
    # if action.l ∉ [0, C.𝒮.A.κ[ceil(s/4)][ceil(a/4)]]
    #     cost += 100.
    # end
    return cost
end

function generate_feedback(state::CASstate,
                          action::CASaction)
    if randn() <= 0.05
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

function compute_level_optimality(C, ℒ)
    total = 0
    lo = 0
    # for s in keys(ℒ.π)
    for (s, state) in enumerate(C.S)
        if terminal(C, state)
            continue
        end
        solve(ℒ, C, s)
        total += 1
        state = C.S[s]
        action = C.A[ℒ.π[s]]
        lo += (action.l == competence(state.state, action.action))
    end
    # println("  ")
    # println(lo)
    # println(total)

    return lo/total
end

function simulate(M::CASSP, L)
    S, A, C = M.S, M.A, M.C
    T_base = deepcopy(M.T)
    c = Vector{Float64}()
    signal_count = 0
    actions_taken = 0
    actions_at_competence = 0
    expected_cost = L.V[M.SIndex[M.s₀]]
    # println("Expected cost to goal: $(ℒ.V[index(state, S)])")
    for i ∈ 1:10
        state = M.s₀
        episode_cost = 0.0
        while true
            s = M.SIndex[state]
            if !haskey(override_rate_records, state)
                override_rate_records[state] = [1 0]
            end
            # println(state, "     ", s)
            a = solve(L, M, s)[1]
            action = A[a]
            actions_taken += 1
            actions_at_competence += (action.l == competence(state.state, action.action))
            # println("Taking action $action in state $state.")
            if action.l == 0 || action.l == 3
                σ = '∅'
            elseif action.l == 1
                σ = generate_feedback(state, action)
                if i == 10
                    y = (σ == '⊕') ? 1 : 0
                    d = hcat(get_state_features(state.state), 1, y)
                    if typeof(state.state) == NodeState
                        record_data(d,joinpath(abspath(@__DIR__), "data", "node_$(action.action.value).csv"))
                    else
                        record_data(d,joinpath(abspath(@__DIR__), "data", "edge_$(action.action.value).csv"))
                    end
                end
            elseif action.l == 2 || (action.l == 1 && !M.flags[M.𝒮.D.SIndex[state.state]][M.𝒮.D.AIndex[action.action]])
                σ = generate_feedback(state, action)
                if i == 10
                    y = (σ == '∅') ? 1 : 0
                    d = hcat(get_state_features(state.state), 2, y)
                    if typeof(state.state) == NodeState
                        record_data(d,joinpath(abspath(@__DIR__), "data", "node_$(action.action.value).csv"))
                    else
                        record_data(d,joinpath(abspath(@__DIR__), "data", "edge_$(action.action.value).csv"))
                    end
                end
            end
            # println("received feedback: $σ")
            if σ != '∅'
                override_rate_records[state][2] += 1
                if i == 10
                    signal_count += 1
                end
            end
            episode_cost += C(M, s, a)
            if σ == '⊖'
                block_transition!(M, state, action)
                state = CASstate(state.state, '∅')
                # M.s₀ = state
                L = solve_model(M)
                continue
            end
            if action.l == 0 || σ == '⊘'
                state = M.S[M.T[s][a][1][1]]
            else
                state = generate_successor(M.𝒮.D, state, action, σ)
            end
            # println(σ, "     | succ state |      ", state)
            if terminal(M, state)
                break
            end
        end

        push!(c, episode_cost)
        M.T = T_base
    end
    println("Total cumulative reward: $(round(mean(c);digits=4)) ⨦ $(std(c))")
    return mean(c), std(c), signal_count, (actions_at_competence / actions_taken), (abs(mean(c) - expected_cost)/expected_cost)
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

    λ = generate_feedback_profile(𝒟, Σ, L)
    ℱ = FeedbackModel(Σ, λ, human_cost)
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
    ℒ = LAOStarSolver(100000, 1000., 1.0, .001, Dict{Integer, Integer}(),
                        zeros(length(C.S)), zeros(length(C.S)),
                        zeros(length(C.S)), zeros(length(C.A)),
                        zeros(Bool, length(C.S)))
    a, total_expanded = solve(ℒ, C, C.SIndex[C.s₀])
    # println("LAO* expanded $total_expanded nodes.")
    # println("Expected cost to goal: $(ℒ.V[C.SIndex[C.s₀]])")
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

override_rate_records = Dict{DomainState, Array{Int}}()

function run_episodes(CAS_vec)
    println("Starting")

    # Tracking information
    los = Vector{Float64}()
    costs = Vector{Float64}()
    stds = Vector{Float64}()
    cost_errors = Vector{Float64}()
    expected_task_costs = Vector{Float64}()
    signal_counts = Vector{Int}()
    lo_function_of_signal_count = Vector{Tuple{Int, Float64}}()
    route_records = Dict{Int, Dict{Tuple{Int, Int}, Vector{Int}}}()
    override_rate_records_by_ep = Vector{Dict{DomainState, Array{Int}}}()
    total_signals_received = 0


    M = build_model()
    C = build_cas(M, [0,1,2,3], ['⊕', '⊖', '⊘', '∅'])
    push!(CAS_vec, deepcopy(C))

    for i=1:500
        ℒ = solve_model(C)
        lo = compute_level_optimality(C, ℒ)

        println(i, "  |  Task: ", M.s₀.id, "→", first(M.G).id, "  |  ", lo)
        push!(los, lo)

        c, std, signal_count, percent_lo, error = simulate(C, ℒ)
        push!(costs, c)
        push!(stds, std)
        push!(cost_errors, error)
        total_signals_received += signal_count
        push!(signal_counts, total_signals_received)
        push!(lo_function_of_signal_count, (total_signals_received, percent_lo))

        update_feedback_profile!(C)
        generate_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)
        update_autonomy_profile!(C, ℒ)
        save_autonomy_profile(C.𝒮.A.κ)

        if i%10 == 0
            route_records[i] = Dict{Tuple{Int,Int}, Vector{Int}}()
            for (init, goal) in test_tasks
                set_route(M, C, init, goal)
                generate_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)
                L = solve_model(C)
                route = get_route(C, L)
                route_records[i][(init, goal)] = route
            end
            push!(override_rate_records_by_ep, deepcopy(override_rate_records))
        end

        set_route(M, C, 12, 7)
        generate_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)
        L = solve_model(C)
        push!(expected_task_costs, L.V[C.SIndex[C.s₀]])

        random_route(M, C)
        generate_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)
    end

    println(costs)
    println(stds)
    println(cost_errors)
    println(los)
    println(lo_function_of_signal_count)
    println(signal_counts)
    println(expected_task_costs)

    x = [i[1] for i in lo_function_of_signal_count]
    y = [i[2] for i in lo_function_of_signal_count]

    g = scatter(x, los, xlabel="Signals Received", ylabel="Level Optimality")
    savefig(g, "level_optimality_by_signal_count.png")

    # g2 = scatter(x, y, xlabel="Signals Received", ylabel="Level Optimality")
    # savefig(g2, "lo_encountered.png")

    g3 = scatter(x, cost_errors, xlabel="Signals Received", ylabel="%Error")
    savefig(g3, "percent_error.png")

    g4 = scatter(x, stds, xlabel="Signals Received", ylabel="Reliability")
    savefig(g4, "percent_error.png")

    g5 = scatter(x=1:300, y=expected_task_costs, xlabel="Episode", ylabel="Expected Cost to Goal")
    savefig(g5, "expected_goal_fixed_tas.png")

    return costs, stds, cost_errors, los, lo_function_of_signal_count, signal_counts, expected_task_costs
end

function get_route(C, L)
    route = Vector{Int}()
    state = C.s₀
    while !(state ∈ C.G)
        push!(route, state.id)
        s = C.SIndex[state]
        a = L.π[s]
        sp = T[s][a][1][1]
        state = C.S[s]
    end
    push!(route, state.id)
end



CAS_vec = Vector{CASSP}()
results2 = run_episodes(CAS_vec)
results = run_episodes(CAS_vec)
init_data()

M = build_model()
C = build_cas(M, [0,1,2,3], ['⊕', '⊖', '⊘', '∅'])
L = solve_model(C)
compute_level_optimality(C, L)
update_autonomy_profile!(C,L)
@show C.𝒮.A.κ[1356][5]
@show C.𝒮.F.λ[1356][5]
generate_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)
solve(L, C, 3201)


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
debug_competence(C, L)
