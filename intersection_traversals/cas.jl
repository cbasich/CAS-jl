struct CASstate
    state::State
        σ::Char
end

struct CASaction
    action::Action
         l::Int
end

struct AutonomyModel
    L::Vector{Int}
    k::Dict{Int, Dict{Int, Int}}
    μ::Function
end

function generate_autonomy_profile(𝒟::DomainModel)
    κ = Dict{Int, Dict{Int, Int}}()
    for (s, state) in enumerate(𝒟.S)
        κ[s] = Dict{Int, Int}()
        for (a, action) in enumerate(𝒟.A)
            κ[s][a] = 1
        end
    end
    return κ
end

function update_potential(ℳ, ℒ, s, a, L)
    state = CASstate(ℳ.𝒮.D.S[s], '∅')
    s2 = ℳ.SIndex[state]
    X = [lookahead(ℒ, ℳ, s2, ((a - 1) * 4 + l + 1) ) for l ∈ L]
    P = 0.75 .* softmax(-1.0 .* X)
    for l = 1:size(L)[1]
        ℳ.potential[s][a][L[l]+1] += P[l]
    end
    clamp!(ℳ.potential[s][a], 0.0, 1.0)
end

function update_autonomy_profile!(ℳ, ℒ)
    κ, S, A = ℳ.𝒮.A.κ, ℳ.𝒮.D.S, ℳ.𝒮.D.A
    for (s, state) in enumerate(S)
        for (a, action) in enumerate(A)
            if κ[s][a] == 3 || κ[s][a] == 0
                continue
            end

            L = [κ[s][a]-1, κ[s][a], κ[s][a]+1]
            update_potential(ℳ, ℒ, s, a, L)

            for i in sortperm(-[ℳ.potential[s][a][l+1] for l in L])
                if rand() <= ℳ.potential[s][a][L[i] + 1]
                    logic()

                    ℳ.𝒮.A.κ[s][a] = L[i]
                    ℳ.potential[s][a][L[i]+1] = 0.0
                    break
                end
            end
        end
    end
end

function competence(state::State,
                   action::Action)
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

function get_state_features(state::State)
    return [state.pos state.occ state.ped state.tra]
end

function generate_feedback_profile(D::DomainModel,
                                   Σ::Vector{Char},
                                   L::Vector{Int})
    λ = Dict{Int, Dict{Int, Dict{Int, Dict{Char, Float64}}}}()
    for (s, state) in enumerate(D.S)
        f = get_state_features(state)
        λ[s] = Dict{Int, Dict{Int, Dict{Char, Float64}}}()
        for (a, action) in enumerate(D.A)
            X, Y = read_data(joinpath(abspath(@__DIR__), "data", ".csv"))
            fm = @formula(y ~ x1 + x2 + x3 + x4)
            logit = lm(fm, hcat(X, Y))

            λ[s][a] = Dict{Int, Dict{Char, Float64}}()
            for l in L
                λ[s][a][l] = Dict{Char, Float64}()
                for σ in Σ
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

function update_feedback_profile!(model)
    λ, 𝒟, Σ, L = C.𝒮.F.λ, C.𝒮.D, C.𝒮.F.Σ, C.𝒮.A.L
    for (s, state) in enumerate(𝒟.S)
        f = get_state_features(state)
        λ[s] = Dict{Int, Dict{Int, Dict{Char, Float64}}}()
        for (a, action) in enumerate(𝒟.A)
            X, Y = read_data(joinpath(abspath(@__DIR__), "data", ".csv"))
            fm = @formula(y ~ x1 + x2 + x3 + x4)
            logit = lm(fm, hcat(X, Y))

            λ[s][a] = Dict{Int, Dict{Char, Float64}}()
            for l in L
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

function human_cost(action::CASaction)
    return [8.0 3. 1. 0.][action.l + 1]
end

##

##

function get_prob_in_horizon(M, s, s′, h, π)
    horizon_probs = dict()
    if s == s′
        horizon_probs[0] = 1.0
    end

    visited = Set((x,1) for x in M.T(s, π[s]))

    while !isempty(visited)
        ((sp, p), i) = pop!(visited)

        if sp == s′
            if !haskey(horizon_probs, i)
                horizon_probs[i] = p
            else
                horizon_probs[i] += p
            end
        end

        if i >= h
            continue
        end

        t = M.T(sp, π[sp])
        for (spp, pp) in t
            push!(visited, ((spp, pp * p), i+1))
        end
    end

    return horizon_probs
end

function δ(s, a)
    return 1.0      # TODO: Replace placeholder function
end

function d(s, a, i)
    return -exp(-δ(s, a) * abs(i))
end

function ξ(M, s, s′, θ, h, π)
    p = 0.
    succ_prob_in_horizon = get_prob_in_horizon(M, s, s′, h, π)
    for i in 1:h
        a = π[s′]
        _λ = M.F.λ[s′][ceil(a/length(M.A.L))][M.A[a]][θ]
        p += succ_prob_in_horizon[i] * d(s, π[s], i) * _λ
    end
    return p
end

function Ξ(M, s, h, π, θ)
    p = 0.
    for s′ in M.S
        p += ξ(M, s, s′, θ, h, π)
    end
    return p
end

##

##

struct CAS
    D::DomainModel
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

function generate_states(𝒮::CAS)
    D, A, F = 𝒮.D, 𝒮.A, 𝒮.F
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

function generate_actions(𝒮::CAS)
    D, A, F = 𝒮.D, 𝒮.A, 𝒮.F
    actions = Vector{CASaction}()
    for action in D.A
        for l in A.L
            new_action = CASaction(action, l)
            push!(actions, new_action)
        end
    end
    return actions
end

function allowed(ℳ, s::Int,
                    a::Int)
    return ℳ.A[a].l <= C.𝒮.A.κ[ceil(s/4)][ceil(a/4)]
end

function generate_transitions!(𝒟, 𝒜, ℱ, ℳ,
                              S::Vector{CASstate},
                              A::Vector{CASaction},
                              G::Set{CASstate})
    T = ℳ.T
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
                p_disapprove = λ[base_s][base_a][1]['⊖']
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

function check_transition_validity(ℳ)
    S, A, T = ℳ.S, ℳ.A, ℳ.T
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

function block_transition!(ℳ::CASSP,
                       state::CASstate,
                      action::CASaction)
    T, L = ℳC.T, ℳ.A.L
    state′ = CASstate(state.state, '⊕')
    s, a = ℳ.SIndex[state], ℳ.AIndex[action]
    for i=1:length(L)
        T[s+i-1][a] = [(s+i-1, 1.0)]
    end
    # T[s+1][a] = [(s+1, 1.0)]
    # T[s+2][a] = [(s+2, 1.0)]
    # T[s+3][a] = [(s+3, 1.0)]
end

function generate_costs(ℳ::CASSP,
                        s::Int,
                        a::Int,)
    D, A, F = ℳ.𝒮.D, ℳ.𝒮.A, ℳ.𝒮.F
    state, action = ℳ.S[s], ℳ.A[a]
    cost = D.C(D, D.SIndex[state.state], D.AIndex[action.action])
    cost += A.μ(state)
    cost += F.ρ(action)
    return cost
end

function build_cas(D::DomainModel,
                   Σ::Vector{Char},
                   L::Vector{Int})
    k = generate_autonomy_profile(D)
    λ = generate_feedback_profile(D, Σ, L)
    A = AutonomyModel(L, κ, autonomy_cost)
    F = FeedbackModel(Σ, λ, human_cost)

    𝒮 = CAS(D, A, F)
    S, s₀, G = generate_states(𝒮)
    A = generate_actions(𝒮)
    T = Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}}()

    ℳ = CASSP(𝒮, S, A, T, generate_costs, s₀, G)
    generate_transitions!(𝒮, ℳ, S, A, G)
    check_transition_validity(ℳ)
    return ℳ
end

##

##

function generate_feedback(state::CASstate,
                          action::CASaction)
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

function compute_level_optimality(ℳ::CASSP, ℒ)
    total = 0
    lo = 0
    for (s, state) in enumerate(C.S)
        if terminal(ℳ, state)
            continue
        end
        solve(ℒ, ℳ, s)
        total += 1
        state = ℳ.S[s]
        action = ℳ.A[ℒ.π[s]]
        lo += (action.l == competence(state.state, action.action))
    end
    return lo/total
end

function solve_model(ℳ::CASSP)
    ℒ = LAOStarSolver(100000, 1000., 1.0, .001, Dict{Integer, Integer}(),
                        zeros(length(ℳ.S)), zeros(length(ℳ.S)),
                        zeros(length(ℳ.S)), zeros(length(ℳ.A)),
                        zeros(Bool, length(ℳ.S)))
    a, total_expanded = solve(ℒ, ℳ, ℳ.SIndex[ℳ.s₀])
    return ℒ
end
