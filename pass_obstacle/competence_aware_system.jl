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
    X = [lookahead(ℒ, C, s2, ((a - 1) * 3 + l + 1) ) for l ∈ L]
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

            # r = rand()
            for i in sortperm(-[C.potential[s][a][l+1] for l in L])
                if rand() <= C.potential[s][a][L[i] + 1]
                    if L[i] == 2
                        if C.𝒮.F.λ[s][a][1]['∅'] < 0.85
                            C.potential[s][a][L[i]+1] = 0.0
                            break
                        end
                    elseif L[i] == 0
                        if C.𝒮.F.λ[s][a][1]['∅'] > 0.35
                            C.potential[s][a][L[i]+1] = 0.0
                            break
                        end
                    elseif L[i] == κ[s][a]
                        C.potential[s][a][L[i]+1] = 0.0
                        break
                    end

                    if L[i] == competence(state, action)
                        println("Updated to competence: ($s, $a) | $(κ[s][a]) | $(L[i])")
                    end

                    C.𝒮.A.κ[s][a] = L[i]
                    C.potential[s][a][L[i]+1] = 0.0
                    # if L[2] == 1 && L[i] == 2
                    #     C.flags[s][a] = true
                    # end
                    break
                end
            end
        end
    end
end

function competence(state::DomainState,
                   action::DomainAction)
    if state.position == 4
        return 2
    end
    if action.value == :stop
        if state.position != 1
            return 0
        elseif state.oncoming < 2 && state.trailing
            return 0
        else
            return 2
        end
    elseif action.value == :edge
        if state.position > 0
            return 0
        else
            return 2
        end
    else
        if state.oncoming == -1
            return 0
        elseif state.oncoming > 1 && !state.priority
            return 0
        elseif state.oncoming > 1 && state.priority
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
        return 12.0
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
    x = [state.position state.oncoming state.trailing state.priority]
    return x
end

function onehot(x)
    return transpose(vcat(Flux.onehot(x[1], 0:4), Flux.onehot(x[2], -1:3), x[3], x[4]))
end

function generate_feedback_profile(𝒟::DomainSSP,
                                   Σ::Vector{Char},
                                   L::Vector{Int})
    λ = Dict{Int, Dict{Int, Dict{Int, Dict{Char, Float64}}}}()
    for (s, state) in enumerate(𝒟.S)
        λ[s] = Dict{Int, Dict{Int, Dict{Char, Float64}}}()
        if state.position == -1
            for (a, action) in enumerate(𝒟.A)
                λ[s][a] = Dict(1 => Dict('∅' => 1.0, '⊘' => 0.0))
            end
            continue
        end
        f = get_state_features(state)
        for (a, action) in enumerate(𝒟.A)
            λ[s][a] = Dict{Int, Dict{Char, Float64}}()
            X, Y = read_data(joinpath(abspath(@__DIR__), "data", "$(action.value).csv"))
            fm = @formula(y ~ x1 + x2 + x3 + x4)
            # logit = lm(fm, hcat(X, Y), contrasts= Dict(:x1 => DummyCoding(), :x2 => DummyCoding()))
            logit = glm(fm, hcat(X, Y), Binomial(), LogitLink())
            for l in [1]
                λ[s][a][l] = Dict{Char, Float64}()
                for σ ∈ Σ
                    # if action != :edge
                    #     f = onehot(f)
                    # end
                    q = DataFrame(f, :auto)
                    if f[1] == -1
                        p = 0.5
                    else
                        p = clamp(predict(logit, q)[1], 0.0, 1.0)
                    end
                    if σ == '∅'
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
        if state.position == -1
            continue
        end
        f = get_state_features(state)
        for (a, action) in enumerate(𝒟.A)
            X, Y = read_data(joinpath(abspath(@__DIR__), "data", "$(action.value).csv"))
            fm = @formula(y ~ x1 + x2 + x3 + x4)
            # logit = lm(fm, hcat(X, Y), contrasts= Dict(:x1 => DummyCoding(), :x2 => DummyCoding()))
            logit = glm(fm, hcat(X, Y), Binomial(), LogitLink())
            for l in [1]
                for σ ∈ Σ
                    # if action != :edge
                    #     f = onehot(f)
                    # end
                    q = DataFrame(f, :auto)
                    if f[1] == -1
                        p = 0.5
                    else
                        p = clamp(predict(logit, q)[1], 0.0, 1.0)
                    end
                    if σ == '∅'
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
    return [10.0 1.0 0.0][action.l+1]
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
    state′ = CASstate(state.state, '⊘')
    s, a = C.SIndex[state], C.AIndex[action]
    T[s][a] = [(s, 1.0)]
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
    # if randn() <= 0.05
    #     if action.l == 1
    #         return ['⊕', '⊖'][rand(1:2)]
    #     elseif action.l == 2
    #         return ['∅', '⊘'][rand(1:2)]
    #     end
    # end
    if state.position == 4
        return '∅'
    end

    if action.value == :stop
        if state.position != 1
            return '⊘'
        elseif state.oncoming < 2 && state.trailing
            return '⊘'
        else
            return '∅'
        end
    elseif action.value == :edge
        if state.position > 0
            return '⊘'
        else
            return '∅'
        end
    else
        if state.oncoming == -1
            return '⊘'
        elseif state.oncoming > 1 && !state.priority
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
    lo = 0
    # for s in keys(ℒ.π)
    #     state = C.S[s]
    for (s, state) in enumerate(C.S)
        if terminal(C, state)
            continue
        end
        solve(ℒ, C, s)
        total += 1
        action = C.A[ℒ.π[s]]
        lo += (action.l == competence(state.state, action.action))
    end

    return lo/total
end

function simulate(M::CASSP, L)
    S, A, C = M.S, M.A, M.C
    c = Vector{Float64}()
    signal_count = 0
    actions_taken = 0
    actions_at_competence = 0
    # println("Expected cost to goal: $(ℒ.V[index(state, S)])")
    for i ∈ 1:1
        state = M.s₀
        episode_cost = 0.0
        while true
            s = M.SIndex[state]
            # println(state, "     ", s)
            a = solve(L, M, s)[1]
            action = A[a]
            actions_taken += 1
            actions_at_competence += (action.l == competence(state.state, action.action))
            # println("Taking action $action in state $state.")
            if action.l == 0 || action.l == 2
                σ = '∅'
            else
                σ = generate_feedback(state.state, action.action)
                y = (σ == '∅')
                d = hcat(get_state_features(state.state), y)
                record_data(d,joinpath(abspath(@__DIR__), "data", "$(action.action.value).csv"))
            end
            # if action.l == 1
            #     σ = generate_feedback(state, action)
            #     y = (σ == '⊕') ? 1 : 0
            #     d = hcat(get_state_features(state.state), 1, y)
            #     record_data(d,joinpath(abspath(@__DIR__), "data", "$(action.action.value).csv"))
            # elseif action.l == 2 || (action.l == 1 && !M.flags[M.𝒮.D.SIndex[state.state]][M.𝒮.D.AIndex[action.action]])
            #     σ = generate_feedback(state, action)
            #     y = (σ == '∅') ? 1 : 0
            #     d = hcat(get_state_features(state.state), 2, y)
            #     record_data(d,joinpath(abspath(@__DIR__), "data", "$(action.action.value).csv"))
            # end
            # println("received feedback: $σ")
            if σ != '∅'
                signal_count += 1
                # println("Received feedback: $σ")
            end
            episode_cost += C(M, s, a)
            # if σ == '⊖'
            #     block_transition!(M, state, action)
            #     state = CASstate(state.state, '∅')
            #     # M.s₀ = state
            #     L = solve_model(M)
            #     continue
            # end
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
    end
    println("Total cumulative reward: $(round(mean(c);digits=4)) ⨦ $(std(c))")
    return mean(c), signal_count, (actions_at_competence / actions_taken)
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
    for action in [:stop, :edge, :go]
        init_pass_obstacle_data(joinpath(abspath(@__DIR__), "data", "$action.csv"))
    end
end

function run_episodes()
    los = Vector{Float64}()
    costs = Vector{Float64}()
    signal_counts = Vector{Int}()
    lo_function_of_signal_count = Vector{Tuple{Int, Float64}}()
    total_signals_received = 0

    M = build_model()
    C = build_cas(M, [0,1,2], ['⊘', '∅'])
    for i=1:2000
        ℒ = solve_model(C)
        lo = compute_level_optimality(C, ℒ)
        println(i, "  |  ", lo)
        push!(los, lo)
        c, signal_count, percent_lo = simulate(C, ℒ)
        push!(costs, c)
        total_signals_received += signal_count
        push!(signal_counts, total_signals_received)
        push!(lo_function_of_signal_count, (total_signals_received, percent_lo))
        update_feedback_profile!(C)
        generate_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)
        update_autonomy_profile!(C, ℒ)
        save_autonomy_profile(C.𝒮.A.κ)
        generate_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)
    end

    println(costs)
    println(los)
    println(lo_function_of_signal_count)
    println(signal_counts)

    x = [i[1] for i in lo_function_of_signal_count]
    y = [i[2] for i in lo_function_of_signal_count]

    g = scatter(x, los, xlabel="Signals Received", ylabel="Level Optimality")
    savefig(g, "PO_level_optimality_by_signal_count.png")

    g2 = scatter(x, y, xlabel="Signals Received", ylabel="Level Optimality")
    savefig(g2, "PO_lo_encountered.png")
end
# M = build_model()
# C = build_cas(M, [0,1,2], ['⊘', '∅'])
# @show C.𝒮.F.λ[10]
# solve_model(M)
# ℒ = solve_model(C)
run_episodes()


init_data()

function debug_competence(C, L)
    κ, λ, D = C.𝒮.A.κ, C.𝒮.F.λ, C.𝒮.D
    total, lo = 0,0
    for (s, state) in enumerate(C.S)
        println("**** $s ****")
        state = C.S[s]
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
            lo += 1
        end
    end
    println(lo/total)
end
debug_competence(C, ℒ)

s = 241
ds = Int(ceil(s/2))
a = 8
da = 2

@show C.potential[ds][da]


State:  CASstate(DomainState(3, 0, true, true, false), '∅')      206 |       Action: CASaction(DomainAction(:go), 1)         8
Competence: 2
Kappa: 1
Lambda: Dict(1 => Dict('⊘' => 0.0937740994336681,'∅' => 0.9062259005663319))

X, Y = read_data(joinpath(abspath(@__DIR__), "data", "stop.csv"))
fm = @formula(y ~ x1 + x2 + x3 + x4)
logit = lm(fm, hcat(X, Y), contrasts= Dict(:x1 => DummyCoding(), :x2 => DummyCoding()))
@show predict(logit, DataFrame(transpose(get_state_features(C.S[331].state)), :auto))
@show transpose(get_state_features(C.S[331].state))

x = [4 3 1 1]
@show d = onehot(x)
@show predict(logit, DataFrame(d, :auto))
@show predict(logit, DataFrame(x, :auto))

glogit = glm(fm, hcat(X, Y), Binomial(), LogitLink())
@show predict(glogit, DataFrame(x, :auto))
