import Base.==

include("maps.jl")
include("../LAOStarSolver.jl")

DIRECTIONS = ['↑', '←', '↓', '→']
function change_direction(θ1, θ2)
    if θ2 == '←'
        if θ1 == '↑'
            return '←'
        elseif θ1 == '←'
            return '↓'
        elseif θ1 == '↓'
            return '→'
        else
            return '↑'
        end
    elseif θ2 == '→'
        if θ1 == '↑'
            return '→'
        elseif θ1 == '→'
            return '↓'
        elseif θ1 == '↓'
            return '←'
        else
            return '↑'
        end
    end
end

struct Graph
    nodes::Dict{Int, Any}
    edges::Dict{Int, Dict{Int, Any}}
end

struct NodeState
    id::Int
    p::Bool
    o::Bool
    v::Int
    θ::Char
end
function NodeState()
    return NodeState(-1, false, false, 0, '↑')
end

struct EdgeState
    u::Int
    v::Int
    θ::Char
    o::Bool
    l::Int
end
function EdgeState()
    return EdgeState(false, -1, -1)
end

DomainState = Union{NodeState, EdgeState}

function ==(a::NodeState, b::NodeState)
    return (isequal(a.id, b.id) && isequal(a.p, b.p) && isequal(a.o, b.o) &&
            isequal(a.v, b.v) && isequal(a.θ, b.θ))
end
function ==(a::EdgeState, b::EdgeState)
    return (isequal(a.u, b.u) && isequal(a.v, b.v) &&
            isequal(a.o, b.o) && isequal(a.l, b.l))
end
function ==(a::NodeState, b::EdgeState)
    return false
end
function ==(a::EdgeState, b::NodeState)
    return false
end

function Base.hash(a::NodeState, h::UInt)
    h = hash(a.id, h)
    h = hash(a.p, h)
    h = hash(a.o, h)
    h = hash(a.v, h)
    h = hash(a.θ, h)
    return h
end

function Base.hash(a::EdgeState, h::UInt)
    h = hash(a.u, h)
    h = hash(a.v, h)
    h = hash(a.o, h)
    h = hash(a.l, h)
    return h
end

struct DomainAction
    value::Char
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
    SIndex::Dict{DomainState, Int}
    AIndex::Dict{DomainAction, Int}
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

function generate_states(G::Graph,
                      init::Int,
                      goal::Int)
    N, E = G.nodes, G.edges
    S = Vector{DomainState}()
    G = Set{DomainState}()
    s₀= PRESERVE_NONE
    g = PRESERVE_NONE
    for node_id in keys(N)
        node = N[node_id]
        for p in [false, true]
            for o in [false, true]
                for v in [0,1,2,3,4]
                    for θ in DIRECTIONS
                        state = NodeState(node_id, p, o, v, θ)
                        push!(S, state)
                        if node_id == goal
                            push!(G, state)
                        end
                    end
                end
            end
        end
    end

    s₀ = NodeState(init, false, false, 0, '↑')

    for u in keys(E)
        for v in keys(E[u])
            for o in [false, true]
                state = EdgeState(u, v, E[u][v]["direction"],
                                     o, E[u][v]["num lanes"])
                push!(S, state)
            end
        end
    end
    return S, s₀, G
end

function terminal(M, state::DomainState)
    return state in M.G
end

function generate_actions()
    A = Vector{DomainAction}()
    for value in ['←', '↑', '→', '⤉']
        push!(A, DomainAction(value))
    end
    return A
end

function generate_transitions!(M, G)
    S, A, T = M.S, M.A, M.T

    for (s, state) in enumerate(S)
        T[s] = Dict{Int, Vector{Pair{Int, Float64}}}()
        if terminal(M, state)
            for (a, action) in enumerate(A)
                T[s][a] = [(s, 1.0)]
            end
        end

        if typeof(state) == NodeState
            for (a, action) in enumerate(A)
                if action.value == '←'
                    T[s][a] = left_turn_distribution(M, state, s, G)
                elseif action.value == '→'
                    T[s][a] = right_turn_distribution(M, state, s, G)
                elseif action.value == '↑'
                    T[s][a] = go_straight_distribution(M, state, s, G)
                else
                    T[s][a] = wait_distribution(M, state, s, G)
                end
            end
        elseif typeof(state) == EdgeState
            for (a, action) in enumerate(A)
                if action.value == '↑'
                    T[s][a] = continue_distribution(M, state, s, G)
                elseif action.value == '⤉'
                    T[s][a] = pass_obstruction_distribution(M, state, s, G)
                else
                    T[s][a] = [(s, 1.0)]
                end
            end
        end
    end
end

function left_turn_distribution(M::DomainSSP, state::DomainState, s::Int, G::Graph)
    T = Vector{Tuple{Int, Float64}}()
    θ′ = change_direction(state.θ, '←')

    N, E = G.nodes, G.edges

    dest_id = N[state.id][θ′]
    if dest_id == -1
        return [(s, 1.0)]
    else
        p = E[state.id][dest_id]["obstruction probability"]

        state′ = EdgeState(state.id, dest_id, θ′, true, E[stat.id][dest_id]["num lanes"])
        push!(T, (M.SIndex[state′], p))

        state′ = EdgeState(state.id, dest_id, θ′, false, E[stat.id][dest_id]["num lanes"])
        push!(T, (M.SIndex[state′], 1-p))
    end
    return T
end

function right_turn_distribution(M::DomainSSP, state::DomainState, s::Int, G::Graph)
    T = Vector{Tuple{Int, Float64}}()
    θ′ = change_direction(state.θ, '→')

    N, E = G.nodes, G.edges

    dest_id = N[state.id][θ′]
    if dest_id == -1
        return [(s, 1.0)]
    else
        state′ = EdgeState(state.id, dest_id, θ′, true, E[stat.id][dest_id]["num lanes"])
        p = E[state.id][dest_id]["obstruction probability"]
        push!(T, (M.SIndex[state′], p))

        state′ = EdgeState(state.id, dest_id, θ′, false, E[stat.id][dest_id]["num lanes"])
        push!(T, (M.SIndex[state′], 1-p))
    end
    return T
end

function go_straight_distribution(M::DomainSSP, state::DomainState, s::Int, G::Graph)
    T = Vector{Tuple{Int, Float64}}()

    N, E = G.nodes, G.edges
    dest_id = N[state.id][state.θ]

    if dest_id == -1
        return [(s, 1.0)]
    else
        state′ = EdgeState(state.id, dest_id, state.θ, true, E[stat.id][dest_id]["num lanes"])
        p = E[state.id][dest_id]["obstruction probability"]
        push!(T, (M.SIndex[state′], p))

        state′ = EdgeState(state.id, dest_id, state.θ, false, E[stat.id][dest_id]["num lanes"])
        push!(T, (M.SIndex[state′], 1-p))
    end
    return T
end

function wait_distribution(M::DomainSSP, state::DomainState, s::Int, G::Graph)
    S = M.S
    T = Vector{Tuple{Int, Float64}}()

    node = G.nodes[state.id]
    p_ped = node["pedestrian probability"]
    p_occl = node["occlusion probability"]
    p_vehicles = node["vehicle probabilities"]

    for num_vehicle in [0,1,2,3]
        state′ = NodeState(state.id, false, false, num_vehicle, state.θ)
        push!(T, (M.SIndex[state′], ((1-p_ped)*(1-p_occl)*p_vehicles[num_vehicle + 1])))
        state′ = NodeState(state.id, true, false, num_vehicle, state.θ)
        push!(T, (M.SIndex[state′], (p_ped*(1-p_occl)*p_vehicles[num_vehicle + 1])))
        state′ = NodeState(state.id, false, true, num_vehicle, state.θ)
        push!(T, (M.SIndex[state′], ((1-p_ped)*p_occl*p_vehicles[num_vehicle + 1])))
        state′ = NodeState(state.id, true, true, num_vehicle, state.θ)
        push!(T, (M.SIndex[state′], (p_ped*p_occl*p_vehicles[num_vehicle + 1])))
    end
    return T
end

function continue_distribution(M::DomainSSP, state::DomainState, s::Int, G::Graph)
    if state.o == true
        return [(s, 1.0)]
    end
    T = Vector{Tuple{Int, Float64}}()

    N, E = G.nodes, G.edges

    edge = E[state.u][state.v]
    p_arrived = (1 / edge["length"]) * (0.5 + 0.5*edge["num lanes"])
    p_driving = 1 - p_arrived
    p_obstruction = edge["obstruction probability"]

    mass = 0.0

    state′ = EdgeState(state.u, state.v, state.θ, true, edge["num lanes"])
    push!(T, (M.SIndex[state′], p_driving * p_obstruction))
    mass += p_driving * p_obstruction
    state′ = EdgeState(state.u, state.v, state.θ, false, edge["num lanes"])
    push!(T, (M.SIndex[state′], p_driving * (1 - p_obstruction)))
    mass += p_driving * (1 - p_obstruction)

    node = N[state.v]
    p_ped = node["pedestrian probability"]
    p_occl = node["occlusion probability"]
    p_vehicles = node["vehicle probabilities"]

    for num_vehicle in [0,1,2,3]
        state′ = NodeState(state.v, false, false, num_vehicle, state.θ)
        push!(T, (M.SIndex[state′], ((1 - mass)*(1-p_ped)*(1-p_occl)*p_vehicles[num_vehicle + 1])))
        state′ = NodeState(state.v, true, false, num_vehicle, state.θ)
        push!(T, (M.SIndex[state′], ((1 - mass)*p_ped*(1-p_occl)*p_vehicles[num_vehicle + 1])))
        state′ = NodeState(state.v, false, true, num_vehicle, state.θ)
        push!(T, (M.SIndex[state′], ((1 - mass)*(1-p_ped)*p_occl*p_vehicles[num_vehicle + 1])))
        state′ = NodeState(state.v, true, true, num_vehicle, state.θ)
        push!(T, (M.SIndex[state′], ((1 - mass)*p_ped*p_occl*p_vehicles[num_vehicle + 1])))
    end

    return T
end

function pass_obstruction_distribution(M::DomainSSP, state::DomainState, s::Int, G::Graph)
    T = Vector{Tuple{Int, Float64}}()

    N, E = G.nodes, G.edges

    if state.o == false
        return [(s, 1.0)]
    else
        num_lanes = E[state.u][state.v]["num lanes"]
        state′ = EdgeState(state.u, state.v, state.θ, false, num_lanes)
        s′ = M.SIndex[state′]
        p = 1.0
        if num_lanes == 1
            p = 0.2
        elseif num_lanes == 2
            p = 0.5
        else
            p = 0.8
        end
        push!(T, (s′, p))
        push!(T, (s, (1-p)))
    end

    return T
end

function generate_costs(M::DomainSSP, s::Int, a::Int)
    if terminal(M, M.S[s])
        return 0.0
    end
    return 1.0
    # M.C = [[1.0 for a=1:length(M.A)] for s=1:length(M.S)]
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
    # G = generate_map(filepath)
    # G = generate_dummy_graph()
    G = generate_ma_graph()
    init = 1
    goal = 7
    S, s₀, goals = generate_states(G, init, goal)
    A = generate_actions()
    T = Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}}()
    M = DomainSSP(S, A, T, generate_costs, s₀, goals)
    generate_transitions!(M, G)
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

function main()
    M = build_model()
    solve_model(M)
end

# main()
