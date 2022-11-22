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

include("MDP.jl")
##
##

struct CASstate
    state::DomainState
    σ::Char
end

struct CASaction
    action::DomainAction
    l::Int
end

##
##  ============= AUTONOMY MODEL ===============

mutable struct AutonomyModel
    L::Vector{Int}
    κ::Dict{Int, Dict{Int, Int}}
    μ::Function
end

function generate_autonomy_profile(D::MDP, L::Vector{Int})
    κ = Dict{Int, Dict{Int, Int}}()
    for (s, state) in enumerate(D.S)
        κ[s] = Dict{Int, Int}()
        for (a, action) in enumerate(D.A)
            κ[s][a] = competence(state, action)
        end
    end
    return κ
end

function competence(state::DomainState, action::DomainAction)
    w = state.w

    if state.position == 4
        return 2
    end

    if w.weather == "snowy" && w.time == "night"
        return 0
    end

    if (w.waiting || w.trailing) && state.priority && action.vaue != :go
        return 0
    end

    if action.value == :stop
        if state.position > 1 || state.oncoming < 1
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
        if (state.oncoming == -1 ||
           (state.position == 0 && state.oncoming == 1 &&
           (w.weather == "rainy" || w.time == "night")))
           return 0
       elseif state.concoming > 1 && !state.priority
           return 0
       else
           return 2
       end
   end
end

function autonomy_cost(state::CASstate)
    if state.σ == '∅'
        return 0.0
    elseif state.σ == '⊘'
        return 10.0
    end
end

##
##  ============== FEEDBACK MODEL ================

mutable struct FeedbackModel
    Σ::Vector{Char}
    λ::Dict{Int, Dict{Int, Dict{Int, Dict{Char, Float64}}}}
    ρ::Function
end

function generate_feedback_profile(D::MDP, Σ::Vector{Char}, L::Vector{Int})
    S, A = D.S, D.A
    λ = Dict(s => Dict(a => Dict(1 => Dict(σ => 0.5 for σ ∈ Σ))
                                                    for a=1:length(A))
                                                    for s=1:length(S))

    for (a, action) in enumerate(A)
        for (s, state) in enumerate(S)
            λ[s][a] = 
        end
    end
end

function human_cost(action::CASaction)
    return [10.0 1.0 0.0][action.l + 1]
end

##
##  ================ CAS ==================

struct CAS
    D::MDP
    A::AUtonomyModel
    F::FeedbackModel
end

mutable struct CASMDP
    𝒮::CAS
    S::Vector{CASstate}
    A::Vector{CASaction}
    T::Vector{Vector{Vector{Float64}}}
    R::Vector{Vector{Float64}}
   s₀::CASstate
    SIndex::Dict{CASstate, Int}
    AIndex::Dict{CASaction, Int}
    blocked::Dict{Int, Dict{Int, Bool}}
end
function CASMDP(𝒮::CAS,
                S::Vector{CASstate},
                A::Vector{CASaction},
                T::Vector{Vector{Vector{Float64}}},
                R::Vector{Vector{Float64}},
               s₀::CASstate)
    SIndex, AIndex = generate_index_dicts(S, A)
    blocked = Dict(s => Dict(a => false for a=1:length(A)) for s=1:length(S))
    return CASMDP(𝒮, S, A, T, C, s₀, G, SIndex, AIndex, blocked)
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

function generate_states(D::MDP, F::FeedbackModel)
    states = Vector{CASstate}()
    for state in D.S
        for σ in F.Σ
            new_state = CASstate(state, σ)
            push!(states, new_state)
        end
    end
    return states, CASstate(D.s₀, '∅')
end

function terminal(C::CASSP, state::CASstate)
    return terminal(state.state) && state.σ == '∅'
end

function generate_actions(D::MDP, A::AutonomyModel)
    actions = Vector{CASaction}()
    for action in D.A
        for l in A.L
            new_action = CASaction(action, l)
            push!(actions, new_action)
        end
    end
    return actions
end

function allowed(C::CAS, s::Int, a::Int)
    return C.A[a].l <= C.𝒮.A.κ[ceil(s/2)][ceil(a/3)] && C.blocked[s][a] == false
end

function generate_transitions!(𝒟::MDP, 𝒜::AutonomyModel, ℱ::FeedbackModel,
                              S::Vector{CASstate}, A::Vector{CASaction}, C)
    T, κ, λ = C.T, 𝒜.κ, ℱ.λ

    for (s, state) in enumerate(S)
        for (a,action) in enumerate(A)
            if terminal(state)
                state′ = CASstate(state.state, '∅')
                sp = C.SIndex[state′]
                T[s][a][sp] = 1.0
                continue
            end

            if state.state.position == -1
                state′ = CASstate(last(𝒟.S), '∅')
                sp = C.SIndex[state′]
                T[s][a][sp] = 1.0
                continue
            end

            base_state = state.state
            base_action = action.action
            base_s = 𝒟.SIndex[base_state]
            base_a = 𝒟.AIndex[base_action]

            if action.l > κ[base_s][base_a]
                T[s][a][s] = 1.0
                continue
            end

            t = 𝒟.T[base_s][base_a]
            if length(t[t .== 1.0]) == 1
                ds = t[t .== 1.0][1]
                dstate = 𝒟.S[sp]
                if dstate.position == -1
                    state′ = CASstate(dstate, '∅')
                    sp = C.SIndex[state′]
                    T[s][a][sp] = 1.0
                end
                if ds == base_s
                    T[s][a][s] = 1.0
                    continue
                end
            end

            if action.l == 0
                state′ = CASstate(DomainState(4, 0, 0, state.state.w), '∅')
                sp = C.SIndex[state′]
                T[s][a][sp] = 1.0
            elseif action.l == 1
                p_override = λ[base_s][base_a][1]['⊘']
                p_null = 1.0 - p_override

                state′ = CASstate(DomainState(4, 0, 0, state.state.w), '⊘')
                sp = C.SIndex[state′]
                T[s][a][sp] = p_override
                for (sp, p) in enumerate(t)
                    T[s][a][(sp-1) * 2 + 1] = p * p_null
                end
            else
                for (sp, p) in enumerate(t)
                    T[s][a][(sp-1) * 2 + 1] = p
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

function block_transition!(C::CASMDP, state::CASstate, action::CASaction)
    state′ = CASstate(state.state, '⊕')
    blocked = Set{Int}()
    s, a = C.SIndex[state′], C.AIndex[action]
    for i=0:15                                                                      ## TODO: Check this '15'
        C.blocked[s+i][a] = true
    end
    return blocked
end

function generate_rewards(C::CASMDP, s::Int, a::Int)
    D, A, F = C.𝒮.D, C.𝒮.A, C.𝒮.F
    state, action = C.S[s], C.A[a]
    reward = D.R[D.SIndex[state.state]][D.AIndex[action.action]]
    reward += A.μ(state)
    reward += F.ρ(action)
    return cost)
end

function generate_feedback(state::DomainState, action::DomainACtion)
    if state.position == 4
        return '∅'
    end

    # Uniformly random feedback under inconsistency
    if rand() <= 0.05
        return ['∅', '⊘'][rand(1:2)]
    end

    w = state.w

    if w.weather == "snowy" && w.time == "night"
        return '⊘'
    end

    if (w.waiting || w.trailing) && state.priority && action.value != :go
        return '⊘'
    end

    # Technical Factors
    if action.value == :stop
        if state.position > 1 || state.oncoming < 1
            return '⊘'
        else
            return '∅'
        end
    elseif action.value == :edge
        if state.position > 0
            return '⊘'
        else
            return '∅''
        end
    else
        if (state.oncoming == -1 ||
           (state.position == 0 && state.oncoming == 1 &&
           (w.weather == "rainy" || w.time == "night")))
            return '⊘'
        elseif state.oncoming > 1 && !state.priority
            return '⊘'
        else
            return '∅'
        end
    end
end

function generate_successor(D::MDP, state::CASstate, action::CASaction, σ::Char)
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

function build_cas(𝒟::MDP, L::Vector{Int}, Σ::Vector{Char})
    κ = generate_autonomy_profile(𝒟, L)
    𝒜 = AutonomyModel(L, κ, autonomy_cost)

    λ = generate_feedback_profile(𝒟, Σ, L)
    ℱ = FeedbackModel(Σ, λ, human_cost)

    𝒮 = CAS(𝒟, 𝒜, ℱ)
    S, s₀ = generate_states(𝒟, ℱ)
    A = generate_actions(𝒟, 𝒜)

    T = [[[0.0 for (i,_) in enumerate(S)]
               for (j,_) in enumerate(A)]
               for (k,_) in enumerate(S)]
    R = generate_costs()

    C = CASMDP(𝒮, S, A, T, R, s₀)
    generate_transitions!(𝒟, 𝒜, ℱ, S, A, C)
    check_transition_validity(C)

    return C
end
