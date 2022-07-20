using Combinatorics
using Statistics
using Random
using DataStructures
import Base.==

include("maps.jl")
include("../LAOStarSolver.jl")

##
##

##TODO: Need to change the 'lanes' feature in world features bc there is a problem
#       with it doubling up on the lanes feature in the edgestate

WorldFeatures = OrderedDict{Symbol, Any}(
    :trailing => [false, true],
    :left_occupied => [false, true],
    :right_occupied => [false, true],
    :waiting => [false, true],
    :lanes => [1, 2, 3],
    :time => ["day", "night"],
    :weather => ["snowy", "rainy", "sunny"],
)

mutable struct WorldState
    trailing::Bool
    left_occupied::Bool
    right_occupied::Bool
    waiting::Bool
    lanes::Int
    time::String
    weather::String
end
function WorldState(T)
    return WorldState(T[1], T[2], T[3], T[4], T[5], T[6], T[7])
end

WorldStates = [WorldState(T) for T in vec(collect(Base.product(values(WorldFeatures)...)))]

function set_world_state!(W1, W2)
    for f in fieldnames(typeof(W1))
        setfield!(W1, f, getproperty(W2, f))
    end
end

function initialize_random_start!(W)
    W.trailing = sample([false true], aweights([0.7, 0.3]))
    W.left_occupied = false
    W.right_occupied = false
    W.waiting = false
    W.lanes = 2
    W.time = sample(WorldFeatures[:time], aweights([0.7, 0.3]))
    W.weather = sample(WorldFeatures[:weather], aweights([0.85, 0.1, 0.05]))
end

function get_random_world_state()
    trailing = sample([false true], aweights([0.7, 0.3]))
    left_occupied = false
    right_occupied = false
    waiting = false
    lanes = 2
    time = sample(WorldFeatures[:time], aweights([0.7, 0.3]))
    weather = sample(WorldFeatures[:weather], aweights([0.85, 0.1, 0.05]))
    return WorldState(trailing, left_occupied, right_occupied, waiting, lanes, time, weather)
end

# function set_random_world_state!(W1)
#
# end
#
# function get_random_world_state()
#     return WorldState()
# end

##
##

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

mutable struct NodeState
    id::Int       # Node ID
    p::Bool       # Boolean existence of pedestrians
    o::Bool       # Boolean existence of occlusion
    v::Int        # Number of blocking vehicles
    θ::Char       # Direction (orientation) of travel
    ISR::Tuple    # Additional features
end
# function NodeState()
#     return NodeState(-1, false, false, 0, '↑')
# end

mutable struct EdgeState
    u::Int        # Node ID of edge origin
    v::Int        # Node ID of edge destination
    θ::Char       # Direction (orientation) of travel
    o::Bool       # Boolean existence of obstruction in road
    l::Int        # Number of lanes on road
    ISR::Tuple    # Additional features
end
# function EdgeState()
#     return EdgeState(false, -1, -1)
# end

DomainState = Union{NodeState, EdgeState}

function ==(a::NodeState, b::NodeState)
    return (isequal(a.id, b.id) &&
            isequal(a.p, b.p) && isequal(a.o, b.o) &&
            isequal(a.v, b.v) && isequal(a.θ, b.θ) && isequal(a.ISR, b.ISR))
end
function ==(a::EdgeState, b::EdgeState)
    return (isequal(a.u, b.u) && isequal(a.v, b.v) &&
            isequal(a.o, b.o) && isequal(a.l, b.l) && isequal(a.ISR, b.ISR))
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
    h = hash(a.ISR, h)
    return h
end
#
function Base.hash(a::EdgeState, h::UInt)
    h = hash(a.u, h)
    h = hash(a.v, h)
    h = hash(a.o, h)
    h = hash(a.l, h)
    h = hash(a.ISR, h)
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

mutable struct DomainSSP
    F_active
    F_inactive
    S
    A
    T
    C
    s₀
    G
    SIndex::Dict{DomainState, Int}
    AIndex::Dict{DomainAction, Int}
    graph
end
function DomainSSP(F_active, F_inactive,
                   S::Vector{DomainState},
                   A::Vector{DomainAction},
                   T::Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}},
                   C::Function,
                  s₀::DomainState,
                   G::Set{DomainState},
               graph::Graph)
    SIndex, AIndex = generate_index_dicts(S, A)
    return DomainSSP(F_active, F_inactive, S, A, T, C, s₀, G, SIndex, AIndex, graph)
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
                      goal::Int,
                      F_active, W)
    N, E = G.nodes, G.edges
    S = Vector{DomainState}()
    G = Set{DomainState}()

    F = OrderedDict(k=>WorldFeatures[k] for k in F_active["node"]
                                        if k in keys(WorldFeatures))
    ISR = vec(collect(Base.product(values(F)...)))
    for node_id in keys(N)
        node = N[node_id]
        for p in [false, true]
            for o in [false, true]
                for v in [0,1,2,3,4]
                    for θ in DIRECTIONS
                        for x in ISR
                            state = NodeState(node_id, p, o, v, θ, x)
                            push!(S, state)
                            if node_id == goal
                                push!(G, state)
                            end
                        end
                    end
                end
            end
        end
    end
    s₀ = NodeState(init, false, false, 0, '↑', Tuple([getproperty(W, f) for
                                     f in F_active["node"] if hasproperty(W, f)]))

    F = OrderedDict(k=>WorldFeatures[k] for k in F_active["edge"]
                                        if k in keys(WorldFeatures))
    ISR = vec(collect(Base.product(values(F)...)))
    for u in keys(E)
        for v in keys(E[u])
            for o in [false, true]
                for x in ISR
                    state = EdgeState(u, v, E[u][v]["direction"],
                                         o, E[u][v]["num lanes"], x)
                    if state ∈ S
                        println(ISR)
                        println(state)
                    end
                    push!(S, state)
                end
            end
        end
    end
    return S, s₀, G
end

function set_init!(M, W, init)
    M.s₀ = NodeState(init, false, false, 0, '↑', Tuple([getproperty(W, f) for
                                    f in M.F_active["node"] if hasproperty(W, f)]))
end

function set_goal!(M, goal)
    F = OrderedDict(k=>WorldFeatures[k] for k in M.F_active["node"]
                                        if k in keys(WorldFeatures))
    ISR = vec(collect(Base.product(values(F)...)))

    M.G = Set{DomainState}()
    for p in [false, true]
        for o in [false, true]
            for v in [0,1,2,3,4]
                for θ in DIRECTIONS
                    for x in ISR
                        state = NodeState(goal, p, o, v, θ, x)
                        push!(M.G, state)
                    end
                end
            end
        end
    end
end

function terminal(M, state::DomainState)
    return state in M.G
end

function generate_actions()
    A = Vector{DomainAction}()
    for value in ['⤉', '↑', '←', '→', '↓']
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
                if action.value == '⤉'
                    T[s][a] = wait_distribution(M, state, s, G)
                else
                    if action.value == '←'
                        T[s][a] = left_turn_distribution(M, state, s, G)
                    elseif action.value == '→'
                        T[s][a] = right_turn_distribution(M, state, s, G)
                    elseif action.value == '↑'
                        T[s][a] = go_straight_distribution(M, state, s, G)
                    elseif action.value == '↓'
                        state′ = NodeState(state.id, state.p, state.o, state.v,
                                           change_direction(state.θ, '↓'), state.ISR)
                        T[s][a] = [(M.SIndex[state′], 1.0)]
                    end
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

        state′ = EdgeState(state.id, dest_id, θ′, true, E[state.id][dest_id]["num lanes"], state.ISR)
        push!(T, (M.SIndex[state′], p))

        state′ = EdgeState(state.id, dest_id, θ′, false, E[state.id][dest_id]["num lanes"], state.ISR)
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
        state′ = EdgeState(state.id, dest_id, θ′, true, E[state.id][dest_id]["num lanes"], state.ISR)
        p = E[state.id][dest_id]["obstruction probability"]
        push!(T, (M.SIndex[state′], p))

        state′ = EdgeState(state.id, dest_id, θ′, false, E[state.id][dest_id]["num lanes"], state.ISR)
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
        state′ = EdgeState(state.id, dest_id, state.θ, true, E[state.id][dest_id]["num lanes"], state.ISR)
        p = E[state.id][dest_id]["obstruction probability"]
        push!(T, (M.SIndex[state′], p))

        state′ = EdgeState(state.id, dest_id, state.θ, false, E[state.id][dest_id]["num lanes"], state.ISR)
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
        state′ = NodeState(state.id, false, false, num_vehicle, state.θ, state.ISR)
        push!(T, (M.SIndex[state′], ((1-p_ped)*(1-p_occl)*p_vehicles[num_vehicle + 1])))
        state′ = NodeState(state.id, true, false, num_vehicle, state.θ, state.ISR)
        push!(T, (M.SIndex[state′], (p_ped*(1-p_occl)*p_vehicles[num_vehicle + 1])))
        state′ = NodeState(state.id, false, true, num_vehicle, state.θ, state.ISR)
        push!(T, (M.SIndex[state′], ((1-p_ped)*p_occl*p_vehicles[num_vehicle + 1])))
        state′ = NodeState(state.id, true, true, num_vehicle, state.θ, state.ISR)
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
    p_arrived = (1 / edge["length"])
    if edge["num lanes"] == 2
        p_arrived *= 2
    elseif edge["num lanes"] == 3
        p_arrived *= 4
    end
    p_driving = 1 - p_arrived
    p_obstruction = edge["obstruction probability"]

    mass = 0.0

    state′ = EdgeState(state.u, state.v, state.θ, true, edge["num lanes"], state.ISR)
    push!(T, (M.SIndex[state′], p_driving * p_obstruction))
    mass += p_driving * p_obstruction
    state′ = EdgeState(state.u, state.v, state.θ, false, edge["num lanes"], state.ISR)
    push!(T, (M.SIndex[state′], p_driving * (1 - p_obstruction)))
    mass += p_driving * (1 - p_obstruction)

    node = N[state.v]
    p_ped = node["pedestrian probability"]
    p_occl = node["occlusion probability"]
    p_vehicles = node["vehicle probabilities"]

    for num_vehicle in [0,1,2,3]
        state′ = NodeState(state.v, false, false, num_vehicle, state.θ, state.ISR)
        push!(T, (M.SIndex[state′], ((1 - mass)*(1-p_ped)*(1-p_occl)*p_vehicles[num_vehicle + 1])))
        state′ = NodeState(state.v, true, false, num_vehicle, state.θ, state.ISR)
        push!(T, (M.SIndex[state′], ((1 - mass)*p_ped*(1-p_occl)*p_vehicles[num_vehicle + 1])))
        state′ = NodeState(state.v, false, true, num_vehicle, state.θ, state.ISR)
        push!(T, (M.SIndex[state′], ((1 - mass)*(1-p_ped)*p_occl*p_vehicles[num_vehicle + 1])))
        state′ = NodeState(state.v, true, true, num_vehicle, state.θ, state.ISR)
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
        state′ = EdgeState(state.u, state.v, state.θ, false, num_lanes, state.ISR)
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
    elseif M.T[s][a] == (s, 1.0)
        return 100.0
    else
        if typeof(M.S[s]) == NodeState
            if M.A[a] == '⤉'
                return 0.1
            else
                return 0.5
            end
        else
            return 0.1
        end
    end
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
    for i ∈ 1:1
        state = M.s₀
        episode_cost = 0.0
        while true
            s = M.SIndex[state]
            a = L.π[s]
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

function update_features!(M, d)
    try
        M.F_inactive = reshape(deleteat!(vec(M.F_inactive), vec(M.F_inactive) .== d), 1, :)
        if d ∉ M.F_active["node"]
            M.F_active["node"] = hcat(M.F_active["node"], d)
        end
        if d ∉ M.F_active["edge"]
            M.F_active["edge"] = hcat(M.F_active["edge"], d)
        end
    catch
        println("Error: discriminator == $d")
    end
end

function build_model(W::WorldState,
                     F_active = Dict("node"=>[:p :o :v],
                                     "edge"=>[:o :l]),
                     F_inactive = reshape([f for f in keys(WorldFeatures)],1,:))
    graph = generate_ma_graph()
    init = rand(1:16)
    goal = rand(1:16)
    while goal == init
        goal = rand(1:16)
    end
    S, s₀, G = generate_states(graph, init, goal, F_active, W)
    A = generate_actions()
    T = Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}}()
    M = DomainSSP(F_active, F_inactive, S, A, T, generate_costs, s₀, G, graph)
    generate_transitions!(M, graph)
    check_transition_validity(M)
    return M
end
function build_model!(M, W)
    S, s₀, G = generate_states(M.graph, M.s₀.id, pop!(M.G).id, M.F_active, W)
    M.S, M.s₀, M.G = S, s₀, G
    SIndex, AIndex = generate_index_dicts(M.S, M.A)
    M.SIndex, M.AIndex = SIndex, AIndex
    M.T = Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}}()
    generate_transitions!(M, M.graph)
    check_transition_validity(M)
end

function solve_model(M::DomainSSP)
    # ℒ = LRTDPsolver(M, 10000., 100, .001, Dict{Int, Int}(),
    #                  false, Set{Int}(), zeros(length(M.S)),
    #                                     zeros(length(M.A)))
    # solve(ℒ, M, M.SIndex[M.s₀])
    # return ℒ
    ℒ = LAOStarSolver(100000, 1000., 1.0, .001, Dict{Integer, Integer}(),
                        zeros(length(M.S)), zeros(length(M.S)),
                        zeros(length(M.S)), zeros(length(M.A)),
                        [false for i=1:length(M.S)])
    a, total_expanded = solve(ℒ, M, M.SIndex[M.s₀])
    println("LAO* expanded $total_expanded nodes.")
    println("Expected cost to goal: $(ℒ.V[M.SIndex[M.s₀]])")
    return ℒ
end
function allowed(D::DomainSSP, s::Int, a::Int)
    return true
end
