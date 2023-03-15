import Base.==

using Statistics

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
    elseif θ2 == '↓'
        if θ1 == '↑'
            return '↓'
        elseif θ1 == '→'
            return '←'
        elseif θ1 == '↓'
            return '↑'
        else
            return '→'
        end
    end
end

struct Graph
    nodes::Dict{Int, Any}
    edges::Dict{Int, Dict{Int, Any}}
end

struct WorldState
    active_avs::Int
    time::String
    weather::String
end
function WorldState(f)
    return WorldState(f[1], f[2], f[3])
end

function generate_random_world_state()
    return WorldState(
        sample(1:4),
        sample(["day", "night"]),
        sample(["sunny", "rainy", "snowy"], aweights([0.85, 0.1, 0.05]))
    )
end

struct NodeState
    id::Int         # Node ID
    p::Bool         # Boolean existence of pedestrians
    o::Bool         # Boolean existence of occlusion
    v::Int          # Number of blocking vehicles
    θ::Char         # Direction (orientation) of travel
    w::WorldState
end
function NodeState()
    return NodeState(-1, false, false, 0, '↑')
end

struct EdgeState
    u::Int          # Node ID of edge origin
    v::Int          # Node ID of edge destination
    θ::Char         # Direction (orientation) of travel
    o::Bool         # Boolean existence of obstruction in road
    l::Int          # Number of lanes on road
    r::String       # Restriction on road segment
    w::WorldState
end
function EdgeState()
    return EdgeState(false, -1, -1)
end

AVState = Union{NodeState, EdgeState}

function ==(a::NodeState, b::NodeState)
    return (isequal(a.id, b.id) && isequal(a.p, b.p) && isequal(a.o, b.o) &&
            isequal(a.v, b.v) && isequal(a.θ, b.θ) && isequal(a.w, b.w))
end
function ==(a::EdgeState, b::EdgeState)
    return (isequal(a.u, b.u) && isequal(a.v, b.v) &&
            isequal(a.o, b.o) && isequal(a.l, b.l) && isequal(a.w, b.w))
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
    h = hash(a.w, h)
    return h
end

function Base.hash(a::EdgeState, h::UInt)
    h = hash(a.u, h)
    h = hash(a.v, h)
    h = hash(a.o, h)
    h = hash(a.l, h)
    h = hash(a.r, h)
    h = hash(a.w, h)
    return h
end

struct AVAction
    value::Char
end

function ==(a::AVAction, b::AVAction)
    return isequal(a.value, b.value)
end

function Base.hash(a::AVAction, h::UInt)
    return hash(a.value, h)
end

mutable struct AVSSP
    S
    A
    T
    C
    s₀
    G
    SIndex::Dict{AVState, Int}
    AIndex::Dict{AVAction, Int}
    graph
end
function AVSSP(S::Vector{AVState},
                   A::Vector{AVAction},
                   T::Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}},
                   C::Array{Array{Float64,1},1},
                  s₀::AVState,
                   G::Set{AVState},
               graph::Graph)
    SIndex, AIndex = generate_index_dicts(S, A)
    return AVSSP(S, A, T, C, s₀, G, SIndex, AIndex, graph)
end

function generate_index_dicts(S::Vector{AVState}, A::Vector{AVAction})
    SIndex = Dict{AVState, Integer}()
    for (s, state) ∈ enumerate(S)
        SIndex[state] = s
    end
    AIndex = Dict{AVAction, Integer}()
    for (a, action) ∈ enumerate(A)
        AIndex[action] = a
    end
    return SIndex, AIndex
end

function generate_av_states(𝒢::Graph,
                      init::Int,
                     goals::Vector{Int},
                         w)
    N, E = 𝒢.nodes, 𝒢.edges
    S = Vector{AVState}()
    G = Set{AVState}()

    W = vec(collect(Base.product(
        1:4, ["day", "night"], ["sunny", "rainy", "snowy"])))

    for node_id in keys(N)
        node = N[node_id]
        for p in [false, true]
            for o in [false, true]
                for v in [0,1,2,3]
                    for θ in DIRECTIONS
                        for w in W
                            state = NodeState(node_id, p, o, v, θ, WorldState(w))
                            push!(S, state)
                            if node_id in goals
                                push!(G, state)
                            end
                        end
                    end
                end
            end
        end
    end

    s₀ = NodeState(init, false, false, 0, '↑', w)

    for u in keys(E)
        for v in keys(E[u])
            for o in [false, true]
                for w in W
                    if haskey(E[u][v], "restriction")
                        state = EdgeState(u, v, E[u][v]["direction"], o,
                        E[u][v]["num lanes"], E[u][v]["restriction"], WorldState(w))
                    else
                        state = EdgeState(u, v, E[u][v]["direction"], o,
                        E[u][v]["num lanes"], "None", WorldState(w))
                    end
                    push!(S, state)
                end
            end
        end
    end
    return S, s₀, G
end

function set_init!(M::AVSSP, init, w)
    M.s₀ = NodeState(init, false, false, 0, '↑', w)
end

function set_goals!(M::AVSSP, goals, w)
    M.G = Set{AVState}()
    for p in [false, true]
        for o in [false, true]
            for v in [0,1,2,3]
                for θ in DIRECTIONS
                    for g in goals
                        state = NodeState(g, p, o, v, θ, w)
                        push!(M.G, state)
                    end
                end
            end
        end
    end
end

function terminal(M::AVSSP, state::AVState)
    return state in M.G
end

function generate_av_actions()
    A = Vector{AVAction}()
    for value in ['←', '↑', '→', '↓', '⤉']
        push!(A, AVAction(value))
    end
    return A
end

function competence(state, action)
    if typeof(state) == EdgeState
        if action.value == '↑'
            if state.r == "None"
                return 2
            else
                return 0
            end
        elseif action.value == '⤉'
            if state.o && state.l == 1
                return 0
            else
                return 2
            end
        else
            return 2
        end
    else
        if action.value == '⤉'
            return 2
        elseif action.value == '→'
            if state.o && state.p && state.v > 1
                return 0
            else
                return 2
            end
        else
            if state.o
                if state.p || state.v > 1
                    return 0
                else
                    return 2
                end
            else
                if state.p && state.v > 2
                    return 0
                else
                    return 2
                end
            end
        end
    end
end

function generate_transitions!(M::AVSSP, G)
    S, A, T = M.S, M.A, M.T

    for (s, state) in enumerate(S)
        T[s] = Dict{Int, Vector{Pair{Int, Float64}}}()
        if terminal(M, state)
            for (a, action) in enumerate(A)
                T[s][a] = [(s, 1.0)]
            end
        end
        w = WorldState(state.w.active_avs + 1, state.w.time, state.w.weather)
        if w.active_avs == 5
            w = WorldState(1, state.w.time, state.w.weather)
        end
        if typeof(state) == NodeState
            for (a, action) in enumerate(A)
                if action.value == '←'
                    T[s][a] = left_turn_distribution(M, state, s, G, w)
                elseif action.value == '→'
                    T[s][a] = right_turn_distribution(M, state, s, G, w)
                elseif action.value == '↑'
                    T[s][a] = go_straight_distribution(M, state, s, G, w)
                elseif action.value == '↓'
                    state′ = NodeState(state.id, state.p, state.o, state.v,
                                       change_direction(state.θ, '↓'), w)
                    T[s][a] = [(M.SIndex[state′], 1.0)]
                else
                    T[s][a] = wait_distribution(M, state, s, G, w)
                end
            end
        elseif typeof(state) == EdgeState
            state′ = EdgeState(state.u, state.v, state.θ,
                               state.o, state.l, state.r, w)
            for (a, action) in enumerate(A)
                if action.value == '↑'
                    T[s][a] = continue_distribution(M, state, s, G, w)
                elseif action.value == '⤉'
                    T[s][a] = pass_obstruction_distribution(M, state, s, G, w)
                else
                    state′ = EdgeState(state.u, state.v, state.θ,
                                       state.o, state.l, state.r, w)
                    T[s][a] = [(M.SIndex[state′], 1.0)]
                end
            end
        end
    end
end

function left_turn_distribution(M::AVSSP, state::AVState, s::Int, G::Graph, w::WorldState)
    T = Vector{Tuple{Int, Float64}}()
    θ′ = change_direction(state.θ, '←')

    N, E = G.nodes, G.edges

    dest_id = N[state.id][θ′]
    if dest_id == -1
        return [(s, 1.0)]
    else
        p = E[state.id][dest_id]["obstruction probability"]
        if haskey(E[state.id][dest_id], "restriction")
            r = E[state.id][dest_id]["restriction"]
        else
            r = "None"
        end
        state′ = EdgeState(state.id, dest_id, θ′, true, E[state.id][dest_id]["num lanes"], r, w)
        push!(T, (M.SIndex[state′], p))

        state′ = EdgeState(state.id, dest_id, θ′, false, E[state.id][dest_id]["num lanes"], r, w)
        push!(T, (M.SIndex[state′], 1-p))
    end
    return T
end

function right_turn_distribution(M::AVSSP, state::AVState, s::Int, G::Graph, w::WorldState)
    T = Vector{Tuple{Int, Float64}}()
    θ′ = change_direction(state.θ, '→')

    N, E = G.nodes, G.edges

    dest_id = N[state.id][θ′]
    if dest_id == -1
        return [(s, 1.0)]
    else
        p = E[state.id][dest_id]["obstruction probability"]
        if haskey(E[state.id][dest_id], "restriction")
            r = E[state.id][dest_id]["restriction"]
        else
            r = "None"
        end
        state′ = EdgeState(state.id, dest_id, θ′, true, E[state.id][dest_id]["num lanes"], r, w)
        push!(T, (M.SIndex[state′], p))

        state′ = EdgeState(state.id, dest_id, θ′, false, E[state.id][dest_id]["num lanes"], r, w)
        push!(T, (M.SIndex[state′], 1-p))
    end
    return T
end

function go_straight_distribution(M::AVSSP, state::AVState, s::Int, G::Graph, w::WorldState)
    T = Vector{Tuple{Int, Float64}}()

    N, E = G.nodes, G.edges
    dest_id = N[state.id][state.θ]

    if dest_id == -1
        return [(s, 1.0)]
    else
        p = E[state.id][dest_id]["obstruction probability"]
        if haskey(E[state.id][dest_id], "restriction")
            r = E[state.id][dest_id]["restriction"]
        else
            r = "None"
        end
        state′ = EdgeState(state.id, dest_id, state.θ, true, E[state.id][dest_id]["num lanes"], r, w)
        push!(T, (M.SIndex[state′], p))

        state′ = EdgeState(state.id, dest_id, state.θ, false, E[state.id][dest_id]["num lanes"], r, w)
        push!(T, (M.SIndex[state′], 1-p))
    end
    return T
end

function wait_distribution(M::AVSSP, state::AVState, s::Int, G::Graph, w::WorldState)
    S = M.S
    T = Vector{Tuple{Int, Float64}}()

    node = G.nodes[state.id]
    p_ped = node["pedestrian probability"]
    p_occl = node["occlusion probability"]
    p_vehicles = node["vehicle probabilities"]

    for num_vehicle in [0,1,2,3]
        state′ = NodeState(state.id, false, false, num_vehicle, state.θ, w)
        push!(T, (M.SIndex[state′], ((1-p_ped)*(1-p_occl)*p_vehicles[num_vehicle + 1])))
        state′ = NodeState(state.id, true, false, num_vehicle, state.θ, w)
        push!(T, (M.SIndex[state′], (p_ped*(1-p_occl)*p_vehicles[num_vehicle + 1])))
        state′ = NodeState(state.id, false, true, num_vehicle, state.θ, w)
        push!(T, (M.SIndex[state′], ((1-p_ped)*p_occl*p_vehicles[num_vehicle + 1])))
        state′ = NodeState(state.id, true, true, num_vehicle, state.θ, w)
        push!(T, (M.SIndex[state′], (p_ped*p_occl*p_vehicles[num_vehicle + 1])))
    end
    return T
end

function continue_distribution(M::AVSSP, state::AVState, s::Int, G::Graph, w::WorldState)
    if state.o == true
        state′  = EdgeState(state.u, state.v, state.θ, state.o, state.l, state.r, w)
        state′′ = EdgeState(state.u, state.v, state.θ, false, state.l, state.r, w)
        return [(M.SIndex[state′], 0.9), (M.SIndex[state′′], 0.1)]
    end
    T = Vector{Tuple{Int, Float64}}()

    N, E = G.nodes, G.edges

    node = N[state.v]
    p_ped = node["pedestrian probability"]
    p_occl = node["occlusion probability"]
    p_vehicles = node["vehicle probabilities"]

    for num_vehicle in [0,1,2,3]
        state′ = NodeState(state.v, false, false, num_vehicle, state.θ, w)
        push!(T, (M.SIndex[state′], ((1-p_ped)*(1-p_occl)*p_vehicles[num_vehicle + 1])))
        state′ = NodeState(state.v, true, false, num_vehicle, state.θ, w)
        push!(T, (M.SIndex[state′], (p_ped*(1-p_occl)*p_vehicles[num_vehicle + 1])))
        state′ = NodeState(state.v, false, true, num_vehicle, state.θ, w)
        push!(T, (M.SIndex[state′], ((1-p_ped)*p_occl*p_vehicles[num_vehicle + 1])))
        state′ = NodeState(state.v, true, true, num_vehicle, state.θ, w)
        push!(T, (M.SIndex[state′], (p_ped*p_occl*p_vehicles[num_vehicle + 1])))
    end

    return T
end

function pass_obstruction_distribution(M::AVSSP, state::AVState, s::Int, G::Graph, w::WorldState)
    T = Vector{Tuple{Int, Float64}}()

    N, E = G.nodes, G.edges

    if state.o == false
        return [(s, 1.0)]
    else
        num_lanes = E[state.u][state.v]["num lanes"]
        state′ = EdgeState(state.u, state.v, state.θ, false, num_lanes, state.r, w)
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

        statew = EdgeState(state.u, state.v, state.θ,
                           state.o, state.l, state.r, w)
        push!(T, (M.SIndex[statew], (1-p)))
    end

    return T
end

function generate_costs(M::AVSSP, s::Int, a::Int)
    if M.S[s] in M.G
        return 0.0
    else
        if typeof(M.S[s]) == NodeState
            if M.A[a].value == '⤉'
                return 1.0
            else
                return 10.0
            end
        else
            e = M.graph.edges[M.S[s].u][M.S[s].v]
            if !M.S[s].o && M.A[a].value == '↑'
                return max(e["length"]/(2^(e["num lanes"] - 1)), 1.0)
            else
                return 1.0
            end
        end
    end
end

function generate_costs!(M::AVSSP)
    for s = 1:length(M.S)
        for a = 1:length(M.A)
            M.C[s][a] = generate_costs(M, s, a)
        end
    end
end

function check_transition_validity(ℳ::AVSSP)
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

function generate_successor(M::AVSSP,
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

function build_av()
    graph = generate_ma_graph()
    init = rand(1:16)
    goals = rand(1:16, 1)
    w = generate_random_world_state()
    while init in goals
        init = rand(1:16)
    end
    S, s₀, G = generate_av_states(graph, init, goals, w)
    A = generate_av_actions()
    T = Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}}()
    C = [[0. for a=1:length(A)] for s=1:length(S)]
    M = AVSSP(S, A, T, C, s₀, G, graph)
    generate_transitions!(M, graph)
    generate_costs!(M)
    check_transition_validity(M)
    return M
end

function solve_model(M::AVSSP)
    ℒ = LAOStarSolver(1000, 1000., 1.0, .001, Dict{Integer, Integer}(),
                        zeros(length(M.S)), zeros(length(M.S)),
                        zeros(length(M.S)), zeros(length(M.A)),
                        [false for i=1:length(M.S)])
    a, total_expanded = solve(ℒ, M, M.SIndex[M.s₀])
    println("LAO* expanded $total_expanded nodes.")
    println("Expected cost to goal: $(ℒ.V[M.SIndex[M.s₀]])")
    return ℒ
end
function allowed(D::AVSSP, s::Int, a::Int)
    return true
end

function quick_sim(M, L)
    state = M.s₀
    while !terminal(M, state)
        s = M.SIndex[state]
        a = solve(L, M, s)[1]
        println("Taking action $(M.A[a]) in state $state")
        state = M.S[generate_successor(M, s, a)]
    end
end
