using Combinatorics
using Statistics
using StatsBase
using Random
using DataStructures
import Base.==

function index(e, S)
    for (i, j) in enumerate(S)
        if e == j
            return i
        end
    end
    return -1
end

include("../LAOStarSolver.jl")

##
##

F_active = [:position :oncoming :priority :trailing :waiting :time :weather]

WorldFeatures = OrderedDict{Symbol, Any}(
    :trailing => [false, true],
    :waiting => [false, true],
    :time => ["day", "night"],
    :weather => ["snowy", "rainy", "sunny"]
)

mutable struct WorldState
    trailing::Bool
    waiting::Bool
    time::String
    weather::String
end
function WorldState(T)
    return WorldState(T[1], T[2], T[3], T[4])
end

function ==(a::WorldState, b::WorldState)
    return (a.trailing == b.trailing && a.waiting == b.waiting &&
            a.time == b.time && a.weather == b.weather)
end

function Base.hash(a::WorldState, h::UInt)
    h = hash(a.trailing, h)
    h = hash(a.waiting, h)
    h = hash(a.time, h)
    h = hash(a.weather, h)
    return h
end

WorldStates = [WorldState(T) for T in vec(collect(Base.product(values(WorldFeatures)...)))]

function set_world_state!(W1, W2)
    for f in fieldnames(typeof(W1))
        setfield!(W1, f, getproperty(W2, f))
    end
end

function set_random_world_state!(W1)
    W.trailing = sample([false true], aweights([0.5, 0.5]))
    W.waiting = sample([false true], aweights([0.6, 0.4]))
    W.time = sample(["night" "day"], aweights([0.6 0.4]))
    W.weather = sample(["sunny" "rainy" "snowy"], aweights([0.5, 0.35, 0.15]))
end

function get_random_world_state()
    return WorldState(sample([false true]), sample([false true]),
                      sample(["night" "day"]), sample(["sunny" "rainy" "snowy"]))
end

##
##

struct DomainState
    position::Int
    oncoming::Int
    priority::Bool
    w::WorldState
end

function ==(a::DomainState, b::DomainState)
    return (isequal(a.position, b.position)
            && isequal(a.oncoming, b.oncoming)
            && isequal(a.priority, b.priority)
            && isequal(a.w, b.w))
end

function Base.hash(a::DomainState, h::UInt)
    h = hash(a.position, h)
    h = hash(a.oncoming, h)
    h = hash(a.priority, h)
    h = hash(a.w, h)
    return h
end

struct DomainAction
    value
end

function ==(a::DomainAction, b::DomainAction)
    return isequal(a.value, b.value)
end

function Base.hash(a::DomainAction, h::UInt)
    return hash(a.value, h)
end

mutable struct MDP
    S::Vector{DomainState}
    A::Vector{DomainAction}
    T::Vector{Vector{Vector{Float64}}}
    R::Vector{Vector{Float64}}
   s₀::DomainState
    G::Set{DomainState}
    SIndex
    AIndex
end
function MDP(S::Vector{DomainState},
             A::Vector{DomainAction},
             T::Vector{Vector{Vector{Float64}}},
             R::Vector{Vector{Float64}},
            s₀::DomainState,
             G::Set{DomainState})
    SIndex, AIndex = generate_index_dicts(S, A)
    return MDP(S, A, T, R, s₀, G, SIndex, AIndex)
end

function generate_index_dicts(S::Vector{DomainState}, A::Vector{DomainAction})
    SIndex = Dict{DomainState, Integer}()
    for (s, state) ∈ enumerate(S)
        SIndex[state] = s
    end
    AIndex = Dict{DomainAction, Integer}()
    for (a, action) ∈ enumerate(A)
        AIndex[action] = a
    end
    return SIndex, AIndex
end

function index(state::DomainState, M::MDP)
    return M.SIndex[state]
end
function index(action::DomainAction, M::MDP)
    return M.AIndex[action]
end

function generate_states(F_active)
    S = Vector{DomainState}()
    G = Set{DomainState}()

    F = OrderedDict(k=>WorldFeatures[k] for k in F_active if k in keys(WorldFeatures))
    W = [WorldState(w) for w in vec(collect(Base.product(values(F)...)))]

    for pos in 0:4
        for onc in -1:3
            if pos < 1 && onc != -1
                continue
            end
            for prio in [false, true]
                for w in W
                    state = DomainState(pos, onc, prio, w)
                    push!(S, state)
                    if pos == 4
                        push!(G, state)
                    end
                end
            end
        end
    end

    for w in W
        push!(S, DomainState(-1, -1, false, w))
    end
    return S, G
end

function terminal(state::DomainState)
    return state.position == 4
end
function terminal(M, state::DomainState)
    return state.position == 4
end

function generate_actions()
    A = Vector{DomainAction}()
    push!(A, DomainAction(:stop))
    push!(A, DomainAction(:edge))
    push!(A, DomainAction(:go))

    return A
end

function generate_transitions(S::Vector{DomainState},
                              A::Vector{DomainAction})
    T = [[[0.0 for (i,_) in enumerate(S)]
               for (j,_) in enumerate(A)]
               for (k,_) in enumerate(S)]

    for (s, state) in enumerate(S)
        if terminal(state)
            for (a, action) in enumerate(A)
                T[s][a][s] = 1.0
            end
            continue
        end
        if ( 2 <= state.position <= 3) && state.oncoming == 3
            state′ = DomainState(-1, -1, false, state.w)
            sp = index(state′, S)
            for (a, action) in enumerate(A)
                T[s][a][sp] = 1.0
            end
            continue
        end
        if state.position == -1
            state′ = DomainState(4, 0, false, state.w)
            sp = index(state′, S)
            for (a, action) in enumerate(A)
                T[s][a][sp] = 1.0
            end
            continue
        end

        for (a, action) in enumerate(A)
            mass = 0.0
            for (sp, state′) in enumerate(S)
                # Impossibilities
                if state.w != state′.w
                    continue
                end
                if action.value == :stop && state.position != state′.position
                    continue
                end
                if state′.position == 0 && state′.oncoming >= 0
                    continue
                end
                if state′.position > 0 && state′.oncoming < 0
                    continue
                end
                if state′.priority && state′.oncoming != 2
                    continue
                end
                if state.oncoming == -1 && state′.priority
                    continue
                end
                if action.value == :stop && state.position != state′.position
                    continue
                end
                if state.oncoming != 2 && state′.priority
                    continue
                end
                if state′.oncoming != 2 && state′.priority
                    continue
                end

                p = 1.0

                # position dynamics
                if action.value == :edge
                    if state.position == 0
                        if state′.position != 1
                            p *= 0.0
                        end
                    elseif state.position == state′.position
                        p *= 0.5
                    elseif state′.position == state.position + 1
                        p *= 0.5
                    else
                        p *= 0.0
                    end
                elseif action.value == :go
                    if state.position == 0
                        if state′.position != 2
                            p *= 0.0
                        end
                    end

                    if state′.position != state.position + 1
                        p *= 0.0
                    end
                end

                # oncoming dynamics
                if state.oncoming == -1 && state′.position > 0
                    if state′.oncoming == 0
                        p *= 0.2
                    elseif state′.oncoming == 1
                        p *= 0.4
                    elseif state′.oncoming == 2
                        p *= 0.3
                    elseif state′.oncoming == 3
                        p *= 0.1
                    else
                        continue
                    end
                elseif state.oncoming == 0 && state′.oncoming == 0
                    p *= 0.8
                elseif state.oncoming == 0 && state′.oncoming == 1
                    p *= 0.2
                elseif state.oncoming == state′.oncoming == 1
                    p *= 0.1
                elseif state.oncoming == 1 && state′.oncoming == 2
                    p *= 0.9
                elseif state.oncoming == 2 && state.priority
                    if state′.oncoming != 2 || !state′.priority
                        continue
                    end
                elseif state.oncoming == 2 && !state.priority
                    if state′.oncoming == 3
                        p *= 0.5
                        if state′.priority
                            continue
                        end
                    elseif state′.oncoming == 2
                        p *= 0.5
                        if !state′.priority
                            continue
                        end
                    else
                        continue
                    end
                elseif state.oncoming == 3
                    if state′.oncoming == 0
                        p *= 0.1
                    elseif state′.oncoming == 1
                        p *= 0.1
                    elseif state′.oncoming == 2
                        p *= 0.7
                    elseif state′.oncoming == 3
                        p *= 0.1
                    end
                else
                    continue
                end

                if p > 0.0
                    mass += p
                    T[s][a][sp] = round(p; digits=5)
                end
            end

            if mass == 0.0
                T[s][a][s] = 1.0
            else
                for p in T[s][a]
                    p = p/mass
                end
            end
        end
    end

    return T
end

function generate_rewards(S::Vector{DomainState},
                          A::Vector{DomainAction})
    R = [[-1.0 for (i,_) in enumerate(A)]
               for (j,_) in enumerate(S)]

    for (s, state) in enumerate(S)
        if state.position == -1
            R[s] *= 100.0
        end
        if terminal(state)
            R[s] *= 0.0
        end
    end

    return R
end

function check_transition_validity(ℳ::MDP)
    S, A, T = ℳ.S, ℳ.A, ℳ.T
    for (s, state) in enumerate(S)
        for (a, action) in enumerate(A)
            mass = 0.0
            for p in T[s][a]
                mass += p
            end
            if round(mass; digits=4) != 1.0
                println("Transition error at state $state and action $action.")
                println("State index: $s      Action index: $a")
                println("Total probability mass of $mass.")
                println("Transition vector is the following: $(T[s][a])")
                # println("Succ state vector: $([S[s] for (s,p) in T[s][a]])")
                @assert false
            end
        end
    end
end

function build_model(W::WorldState)
    S, goals = generate_states(F_active)
    s₀ = DomainState(0, -1, false, WorldState(Tuple([getproperty(W, f)
                    for f in F_active if hasproperty(W, f)])))
    A = generate_actions()
    T = generate_transitions(S,A)
    R = generate_rewards(S,A)
    M = MDP(S, A, T, R, s₀, goals)
    check_transition_validity(M)
    return M
end
function build_model!(M, W)
    M.S, M.G = generate_states(M.F_active)
    M.s₀ = DomainState(0, -1, false, Tuple([getproperty(W, f) for f in M.F_active if hasproperty(W, f)]))
    M.SIndex, M.AIndex = generate_index_dicts(M.S, M.A)
    M.T = Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}}()
    generate_transitions!(M)
    check_transition_validity(M)
end

function solve_model(M::MDP)
    ℒ = LAOStarSolver(100000, 1000., 1.0, .001, Dict{Integer, Integer}(),
                        zeros(length(M.S)), zeros(length(M.S)),
                        zeros(length(M.S)), zeros(length(M.A)))
    a, total_expanded = solve(ℒ, M, M.SIndex[M.s₀])
    println("LAO* expanded $total_expanded nodes.")
    println("Expected cost to goal: $(ℒ.V[M.SIndex[M.s₀]])")
    return ℒ
end

function generate_successor(M::MDP,
                            s::Integer,
                            a::Integer)::Integer
    thresh = rand()
    p = 0.
    T = M.T[s][a]
    for (s′, prob) ∈ T
        p += prob
        if p >= thresh
            return s′
        end
    end
end

function simulate(M::MDP, L)
    S, A, C = M.S, M.A, M.C
    c = Vector{Float64}()
    # println("Expected cost to goal: $(ℒ.V[index(state, S)])")
    for i ∈ 1:100
        state = M.s₀
        episode_cost = 0.0
        while true
            s = M.SIndex[state]
            a = L.π[s]
            action = A[a]
            # println("Taking action $action in state $state.")
            episode_cost += C(M, s, a)
            state = S[generate_successor(M, s, a)]

            if terminal(M, state)
                break
            end
        end

        push!(c, episode_cost)
    end
    println("Total cumulative reward: $(round(mean(c);digits=4)) ⨦ $(std(c))")
end

# function main()
#     println("start")
#     M = build_model()
#     L = solve_model(M)
#     simulate(M, L)
# end
#
# main()
