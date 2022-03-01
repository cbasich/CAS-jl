using Combinatorics
using Statistics
using Random
import Base.==

include("../LAOStarSolver.jl")

struct DomainState
    position::Int
    oncoming::Int
    trailing::Bool
    dynamic::Bool
    priority::Bool
end

function ==(a::DomainState, b::DomainState)
    return (isequal(a.position, b.position)
            && isequal(a.oncoming, b.oncoming) && isequal(a.trailing, b.trailing)
            && isequal(a.dynamic, b.dynamic) && isequal(a.priority, b.priority))
end

function Base.hash(a::DomainState, h::UInt)
    h = hash(a.position, h)
    h = hash(a.oncoming, h)
    h = hash(a.trailing, h)
    h = hash(a.dynamic, h)
    h = hash(a.priority, h)
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

struct DomainSSP
    S
    A
    T
    C
   s₀
    G
    SIndex
    AIndex
end
function DomainSSP(S::Vector{DomainState},
                   A::Vector{DomainAction},
                   T::Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}},
                   C::Function,
                  s₀::DomainState,
                   G::Set{DomainState})
    SIndex, AIndex = generate_index_dicts(S, A)
    return DomainSSP(S, A, T, C, s₀, G, SIndex, AIndex)
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

function index(state::DomainState, M::DomainSSP)
    return M.SIndex[state]
end
function index(action::DomainAction, M::DomainSSP)
    return M.AIndex[action]
end

function generate_states()
    S = Vector{DomainState}()
    G = Set{DomainState}()

    for pos in 0:4
        for o in -1:3
            if pos < 1 && o != -1
                continue
            end
            for t in [false, true]
                for d in [false, true]
                    for p in [false, true]
                        state = DomainState(pos, o, t, d, p)
                        push!(S, state)
                        if pos == 4
                            push!(G, state)
                        end
                    end
                end
            end
        end
    end

    push!(S, DomainState(-1, -1, false, false, false))
    return S, G
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

function generate_transitions!(M::DomainSSP)
    S, A, T = M.S, M.A, M.T

    for (s, state) in enumerate(S)
        T[s] = Dict{Int, Vector{Pair{Int, Float64}}}()

        if terminal(M, state)
            for (a, action) in enumerate(A)
                T[s][a] = [(s, 1.0)]
            end
            continue
        end
        if ( 2 <= state.position <= 3) && state.oncoming == 3
            state′ = DomainState(-1, -1, false, false, false)
            for (a, action) in enumerate(A)
                T[s][a] = [(M.SIndex[state′], 1.0)]
            end
            continue
        end
        if state.position == -1
            state′ = DomainState(4, 0, false, false, false)
            for (a, action) in enumerate(A)
                T[s][a] = [(M.SIndex[state′], 1.0)]
            end
            continue
        end

        for (a, action) in enumerate(A)
            T[s][a] = Vector{Pair{Int, Float64}}()
            mass = 0.0
            for (sp, state′) in enumerate(S)
                # Impossibilities
                if state′.dynamic != state.dynamic
                    continue
                end
                if state.trailing && !state′.trailing
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

                # Trailing dynamics
                if state.trailing == state′.trailing == false
                    p *= 0.8
                elseif !state.trailing && state′.trailing
                    p *= 0.2
                elseif state.trailing == state′.trailing == true
                    p *= 1.0
                end

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
                    push!(T[s][a], (sp, round(p; digits=5)))
                end
            end

            if length(T[s][a]) == 0
                push!(T[s][a], (s, 1.0))
            else
                for (s, p) in T[s][a]
                    p = p/mass
                end
            end
        end
    end
end

function generate_costs(M::DomainSSP, s::Int, a::Int)
    if M.S[s].position == -1
        return 100.0
    end

    if terminal(M, M.S[s])
        return 0.0
    end

    return 1.0
end

function check_transition_validity(ℳ::DomainSSP)
    S, A, T = ℳ.S, ℳ.A, ℳ.T
    for (s, state) in enumerate(S)
        for (a, action) in enumerate(A)
            mass = 0.0
            for (s′, p) in T[s][a]
                mass += p
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

function build_model()
    S, goals = generate_states()
    s₀ = DomainState(0, -1, false, false, false)
    A = generate_actions()
    T = Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}}()
    M = DomainSSP(S, A, T, generate_costs, s₀, goals)
    generate_transitions!(M)
    check_transition_validity(M)
    return M
end

function solve_model(M::DomainSSP)
    ℒ = LAOStarSolver(100000, 1000., 1.0, .001, Dict{Integer, Integer}(),
                        zeros(length(M.S)), zeros(length(M.S)),
                        zeros(length(M.S)), zeros(length(M.A)))
    a, total_expanded = solve(ℒ, M, M.SIndex[M.s₀])
    println("LAO* expanded $total_expanded nodes.")
    println("Expected cost to goal: $(ℒ.V[M.SIndex[M.s₀]])")
    return ℒ
end

function generate_successor(M::DomainSSP,
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

function simulate(M::DomainSSP, L)
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
