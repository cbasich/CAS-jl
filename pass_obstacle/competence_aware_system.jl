import Combinatorics

using GLM
using DataFrames
using CSV
using JLD

# include("utils.jl")
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

function generate_autonomy_profile(𝒟::DomainSSP,
                                   L::Vector{Int})
    κ = Dict{Int, Dict{Int, Int}}()
    for (s, state) in enumerate(𝒟.S)
        κ[s] = Dict{Int, Int}()
        for (a, action) in enumerate(𝒟.A)
            if state.oncoming == 0
                κ[s][a] = 3
            elseif state.oncoming == -1 && action == :go
                κ[s][a] = 1
            elseif state.position > 1
                κ[s][a] = 2
            else
                κ[s][a] = 1
            end
        end
    end
    return κ
end

function update_potential(C, ℒ, s, a, L)
    state = CASstate(C.𝒮.D.S[s], '∅')
    s2 = C.SIndex[state]
    X = [lookahead(ℒ, C, s2, ((a - 1) * 4 + l + 1) ) for l ∈ L]
    P = softmax(-1.0 .* X)
    for l=1:size(L)[1]
        C.potential[s][a][L[l]+1] += .75 .* P[l]
    end
    clamp!(C.potential[s][a], 0.0, 1.0)
end

function update_autonomy_profile!(C, ℒ)
    κ = C.𝒮.A.κ
    for (s, state) in enumerate(C.𝒮.D.S)
        for (a, action) in enumerate(C.𝒮.D.A)
            if κ[s][a] == 3 || κ[s][a] == 0
                continue
            end

            L = [κ[s][a]-1, κ[s][a], κ[s][a]+1]
            update_potential(C, ℒ, s, a, L)

            r = randn()
            for i in sortperm(-[C.potential[s][a][l+1] for l in L])
                if r <= C.potential[s][a][i]
                    if L[i] == 3
                        if C.𝒮.F.λ[s][a][2]['∅'] < 0.85
                            break
                        end
                    elseif L[i] == 0
                        if C.𝒮.F.λ[s][a][1]['⊕'] > 0.35
                            break
                        end
                    end

                    κ[s][a] = L[i]
                    C.potential[s][a][L[i]+1] = 0.0
                    if L[2] == 1 && L[i] == 2
                        C.flags[s][a] = true
                    end
                end
            end
        end
    end
end

function competence(state::DomainState,
                   action::DomainAction)
    if state.position < 2 && state.oncoming > 1 && !state.priority
        return 0
    elseif state.position < 2 && state.trailing
        return 0
    elseif state.position >= 1 && state.oncoming == 2 && !state.priority
        return 0
    else
        return 3
    end
end

function save_autonomy_profile(κ)
    save(joinpath(abspath(@__DIR__), "params.jld"), "κ", κ)
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

function get_state_features(state::DomainState)
    return [state.position state.oncoming state.trailing state.dynamic state.priority]
end

function generate_feedback_profile(𝒟::DomainSSP,
                                   Σ::Vector{Char},
                                   L::Vector{Int})
    λ = Dict{Int, Dict{Int, Dict{Int, Dict{Char, Float64}}}}()
    for (s, state) in enumerate(𝒟.S)
        f = get_state_features(state)
        λ[s] = Dict{Int, Dict{Int, Dict{Char, Float64}}}()
        for (a, action) in enumerate(𝒟.A)
            λ[s][a] = Dict{Int, Dict{Char, Float64}}()
            X, Y = read_data(joinpath(abspath(@__DIR__), "data", "$(action.value).csv"))
            fm = @formula(y ~ x1 + x2 + x3 + x4 + x5 + x6)
            logit = lm(fm, hcat(X, Y), contrasts= Dict(:x1 => DummyCoding(), :x2 => DummyCoding()))
            logit = glm(fm, hcat(X, Y), Binomial(), ProbitLink())
            for l in [1,2]
                λ[s][a][l] = Dict{Char, Float64}()
                for σ ∈ Σ
                    q = DataFrame(hcat(f, l), :auto)
                    p = predict(logit, q)[1]
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
            λ[s][a] = Dict{Int, Dict{Char, Float64}}()
            X, Y = read_data(joinpath(abspath(@__DIR__), "data", "$(action.value).csv"))
            fm = @formula(y ~ x1 + x2 + x3 + x4 + x5 + x6)
            logit = glm(fm, hcat(X, Y), Binomial(), ProbitLink())
            for l in [1,2]
                λ[s][a][l] = Dict{Char, Float64}()
                for σ ∈ Σ
                    q = DataFrame(hcat(f, l), :auto)
                    p = predict(logit, q)[1]
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
    return [10.0 1.0 1.0 0.0][action.l+1]
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
            if t == [(base_s, 1.0)] || action.l ∉ [0, κ[base_s][base_a]]
                T[s][a] = [(s, 1.0)]
                continue
            end

            T[s][a] = Vector{Tuple{Int, Float64}}()
            if action.l == 0
                state′ = CASstate(DomainState(4, 0, 0, state.state.dynamic, 0), '∅')
                push!(T[s][a], (C.SIndex[state′], 1.0))
            elseif action.l == 1
                p_approve = ℱ.λ[base_s][base_a][1]['⊕']
                p_disapprove = 1.0 - p_approve
                push!(T[s][a], ((base_s-1) * 4 + 2, p_disapprove))
                for (sp, p) in t
                    push!(T[s][a], ((sp-1) * 4 + 1, p * p_approve))
                end
            elseif action.l == 2
                p_override = ℱ.λ[base_s][base_a][2]['⊘']
                p_null = 1.0 - p_override
                state′ = CASstate(DomainState(4, 0, 0, state.state.dynamic, 0), '⊘')
                push!(T[s][a], (C.SIndex[state′], p_override))
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

function block_transitioN!(C::CASSP,
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
    return cost
end

function generate_feedback(state::CASstate,
                          action::CASaction)
    # if randn() <= 0.05
    #     if action.l == 1
    #         return ['⊕', '⊖'][rand(1:2)]
    #     elseif action.l == 2
    #         return ['∅', '⊘'][rand(1:2)]
    #     end
    # end

    if state.position < 2 && state.oncoming > 1 && !state.priority
        return (action.l == 1) ? '⊖' : '⊘'
    elseif state.position < 2 && state.trailing
        return (action.l == 1) ? '⊖' : '⊘'
    elseif state.position >= 1 && state.oncoming == 2 && !state.priority
        return (action.l == 1) ? '⊖' : '⊘'
    else
        return (action.l == 1) ? '⊕' : '⊖'
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
    for s in keys(ℒ.π)
        total += 1
        state = C.S[s]
        action = C.A[ℒ.π[s]]
        lo += (action.l == competence(state.state, action.action))
    end

    return lo/total
end

function simulate(M::CASSP, L)
    S, A, C = M.S, M.A, M.C
    c = Vector{Float64}()
    # println("Expected cost to goal: $(ℒ.V[index(state, S)])")
    for i ∈ 1:1
        state = M.s₀
        episode_cost = 0.0
        while true
            s = M.SIndex[state]
            # println(state, "     ", s)
            a = L.π[s]
            action = A[a]
            println("Taking action $action in state $state.")
            if action.l == 0 || action.l == 3
                σ = '∅'
            elseif action.l == 1
                σ = generate_feedback(state, action)
                y = (σ == '⊕') ? 1 : 0
                d = hcat(get_state_features(state.state), 1, y)
                if typeof(state.state) == NodeState
                    record_data(d,joinpath(abspath(@__DIR__), "data", "node_$(action.action.value).csv"))
                else
                    record_data(d,joinpath(abspath(@__DIR__), "data", "edge_$(action.action.value).csv"))
                end
            elseif action.l == 2 || (action.l == 1 && !M.flags[M.𝒮.D.SIndex[state.state]][M.𝒮.D.AIndex[action.action]])
                σ = generate_feedback(state, action)
                y = (σ == '∅') ? 1 : 0
                d = hcat(get_state_features(state.state), 2, y)
                if typeof(state.state) == NodeState
                    record_data(d,joinpath(abspath(@__DIR__), "data", "node_$(action.action.value).csv"))
                else
                    record_data(d,joinpath(abspath(@__DIR__), "data", "edge_$(action.action.value).csv"))
                end
            end
            println("received feedback: $σ")
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
            println(σ, "     | succ state |      ", state)
            if terminal(M, state)
                break
            end
        end

        push!(c, episode_cost)
    end
    println("Total cumulative reward: $(round(mean(c);digits=4)) ⨦ $(std(c))")
    return mean(c)
end


function build_cas(𝒟::DomainSSP,
                   L::Vector{Int},
                   Σ::Vector{Char})
    if ispath(joinpath(abspath(@__DIR__), "params"))
        κ = load_autonomy_profile()
    else
        κ = generate_autonomy_profile(𝒟, L)
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
    for action in ["stop", "edge", "go"]
        init_pass_obstacle_data(joinpath(abspath(@__DIR__), "data", "$action.csv"))
    end
end

function run_episodes()
    M = build_model()
    C = build_cas(M, [0,1,2,3], ['⊕', '⊖', '⊘', '∅'])

    los = Vector{Float64}()
    costs = Vector{Float64}()
    for i=1:1
        println(i)
        ℒ = solve_model(C)
        push!(los, compute_level_optimality(C, ℒ))
        push!(costs, simulate(C, ℒ))
        update_feedback_profile!(C)
        generate_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)
        update_autonomy_profile!(C, ℒ)
        save_autonomy_profile(C.𝒮.A.κ)
    end

    println(costs)
    println(los)
end
M = build_model()
C = build_cas(M, [0,1,2,3], ['⊕', '⊖', '⊘', '∅'])
@show C.𝒮.F.λ[10]
solve_model(M)
ℒ = solve_model(C)
run_episodes()


init_data()
@show M.G
@show C.G
