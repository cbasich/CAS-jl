import Base.==

include("../../LAOStarSolver.jl")


function index(X, x)
    for i = 1:length(X)
        if x == X[i]
            return i
        end
    end
    return -1
end

positions = [:approaching, :at, :in, :through, :after]
traffics = [:empty, :light, :busy]

struct State
    pos::Symbol
    occ::Bool
    ped::Bool
    tra::Symbol
end

struct Action
    value::Symbol
end

mutable struct DomainSSP
    S::Vector{State}
    A::Vector{Action}
    T::Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}}
    C::Function
    s₀::State
    G::Set{State}
    SIndex::Dict{State, Int}
    AIndex::Dict{Action, Int}
end
function DomainSSP(S::Vector{State},
                   A::Vector{Action},
                   T::Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}},
                   C::Function,
                  s₀::State,
                   G::Set{State})
    SIndex, AIndex = generate_index_dicts(S, A)
    return DomainSSP(S, A, T, C, s₀, G, SIndex, AIndex)
end

function generate_index_dicts(S::Vector{State}, A::Vector{Action})
    SIndex = Dict{State, Int}()
    for (s, state) in enumerate(S)
        SIndex[state] = s
    end
    AIndex = Dict{Action, Int}()
    for (a, action) in enumerate(A)
        AIndex[action] = a
    end
    return SIndex, AIndex
end

function generate_states()
    S = Vector{State}()
    G = Set{State}()

    for pos in positions
        for occ in 0:1
            for ped in 0:1
                for tra in traffics
                    push!(S, State(pos, occ, ped, tra))
                    if pos == :after
                        push!(G, State(pos, occ, ped, tra))
                    end
                end
            end
        end
    end
    return S, G
end

function generate_actions()
    A = Vector{Action}()
    push!(A, Action(:stop))
    push!(A, Action(:edge))
    push!(A, Action(:go))
end

function generate_init(p, o, v)
    if v == 0
        return State(:approaching, o, p, :empty)
    elseif v < 4
        return State(:approaching, o, p, :light)
    else
        return State(:approaching, o, p, :busy)
    end
end

function terminal(M, state::State)
    return state in M.G
end

function generate_transitions!(M)
    S, A, T = M.S, M.A, M.T

    for (s, state) in enumerate(S)
        T[s] = Dict{Int, Vector{Pair{Int, Float64}}}()
        if terminal(M, state)
            for (a, action) in enumerate(A)
                T[s][a] = [(s, 1.0)]
            end
        end

        for (a, action) in enumerate(A)
            if action.value == :stop
                T[s][a] = stop_distribution(M, state, s)
            elseif action.value == :edge
                T[s][a] = edge_distribution(M, state, s)
            else
                T[s][a] = go_distribution(M, state, s)
            end
        end
    end
end

function stop_distribution(M::DomainSSP, state::State, s::Int)
    T = Vector{Tuple{Int, Float64}}()

    for (s′, state′) in enumerate(M.S)
        p = 1.
        if state.pos != state′.pos || state.occ != state′.occ
            continue
        else
            if state.tra == :empty
                if state′.tra == :empty || state′.tra == :light
                    p *= 0.5
                else
                    continue
                end
            elseif state.tra == :light
                if state′.tra == :empty
                    p *= 0.25
                elseif state′.tra == :light
                    p *= 0.5
                else
                    p *= 0.25
                end
            elseif state.tra == :busy
                if state′.tra == :empty
                    continue
                elseif state′.tra == :light
                    p *= 0.5
                else
                    p *= 0.5
                end
            end

            if state.ped == state′.ped
                p *= 0.75
            else
                p *= 0.25
            end
        end

        push!(T, (s′, p))
    end
    return T
end

function edge_distribution(M::DomainSSP, state::State, s::Int)
    T = Vector{Tuple{Int, Float64}}()

    if !(state.pos == :at || state.pos == :in)
        push!(T, (s, 1.0))
        return T
    end

    for (s′, state′) in enumerate(M.S)
        p = 1.
        if state.pos == :at
            if state′.pos != :in
                continue
            end
        elseif state.pos == :in
            if state′.pos == :in
                p *= 0.5
            elseif state′.pos == :through
                p *= 0.5
            else
                continue
            end
        else
            continue
        end
        if state.tra == :empty
            if state′.tra == :empty || state′.tra == :light
                p *= 0.5
            else
                continue
            end
        elseif state.tra == :light
            if state′.tra == :empty
                p *= 0.25
            elseif state′.tra == :light
                p *= 0.5
            else
                p *= 0.25
            end
        else
            if state′.tra == :empty
                continue
            elseif state′.tra == :light
                p *= 0.5
            else
                p *= 0.5
            end
        end

        if state.ped == state′.ped
            p *= 0.75
        else
            p *= 0.25
        end

        if state′.pos == :through || state′.pos == :after
            if state′.occ
                continue
            end
        elseif state′.pos == :in
            if state.occ && !state′.occ
                p *= 0.85
            elseif state.occ && state′.occ
                p *= 0.15
            elseif !state.occ && state′.occ
                continue
            end
        else
            if state.occ != state′.occ
                continue
            end
        end
        push!(T, (s′, p))
    end
    return T
end

function go_distribution(M::DomainSSP, state::State, s::Int)
    T = Vector{Tuple{Int, Float64}}()

    if state.pos == :at && (state.occ || state.tra != :empty)
        push!(T, (s, 1.0))
        return T
    end

    for (s′, state′) in enumerate(M.S)
        p = 1.
        if (index(positions, state′.pos) != index(positions, state.pos) + 1 &&
            !(state.pos == state′.pos == :after))
            continue
        else
            if state.tra == :empty
                if state′.tra == :empty || state′.tra == :light
                    p *= 0.5
                else
                    continue
                end
            elseif state.tra == :light
                if state′.tra == :empty
                    p *= 0.25
                elseif state′.tra == :light
                    p *= 0.5
                else
                    p *= 0.25
                end
            else
                if state′.tra == :empty
                    continue
                elseif state′.tra == :light
                    p *= 0.5
                else
                    p *= 0.5
                end
            end

            if state.ped == state′.ped
                p *= 0.75
            else
                p *= 0.25
            end

            if state′.pos == :through || state′.pos == :after
                if state′.occ
                    continue
                end
            elseif state′.pos == :in
                if state.occ && !state′.occ
                    p *= 0.85
                elseif state.occ && state′.occ
                    p *= 0.15
                elseif !state.occ && state′.occ
                    continue
                end
            else
                if state.occ != state′.occ
                    continue
                end
            end
        end

        push!(T, (s′, p))
    end
    return T
end

function generate_costs(M::DomainSSP, s::Int, a::Int)
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
                println("Succ state vector: $([(S[s],p) for (s,p) in T[s][a]])")
                @assert false
            end
        end
    end
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

function simulate(M::DomainSSP, solver)
    S, A, C = M.S, M.A, M.C
    c = Vector{Float64}()
    # println("Expected cost to goal: $(ℒ.V[index(state, S)])")
    for i ∈ 1:1
        state = M.s₀
        episode_cost = 0.0
        while true
            s = M.SIndex[state]
            a = solver.π[s]
            action = A[a]
            println("Taking action $action in state $state.")
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

function build_model()
    S, G = generate_states()
    A = generate_actions()
    T = Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}}()
    s₀ = generate_init(1, 1, 3)
    M = DomainSSP(S, A, T, generate_costs, s₀, G)
    generate_transitions!(M)
    check_transition_validity(M)
    return M
end

function solve_model(M::DomainSSP)
    ℒ = LAOStarSolver(100000, 1000., 1.0, .001, Dict{Integer, Integer}(),
                        zeros(length(M.S)), zeros(length(M.S)),
                        zeros(length(M.S)), zeros(length(M.A)),
                        zeros(Bool, length(M.S)))
    a, total_expanded = solve(ℒ, M, M.SIndex[M.s₀])
    println("LAO* expanded $total_expanded nodes.")
    println("Expected cost to goal: $(ℒ.V[M.SIndex[M.s₀]])")
    return ℒ
end
function allowed(D::DomainSSP, s::Int, a::Int)
    return true
end

M = build_model()
L = solve_model(M)
