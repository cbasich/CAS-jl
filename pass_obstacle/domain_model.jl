import Base.==

include("../LAOStarSovler.jl")

struct DomainState
    pos::Int
    trailing::Bool
    oncoming::Bool
    dynamic::Bool
    stopped::Bool
end

function ==(a::DomainState, b::DomainState)
    return (isequal(a.lanes, b.lanes) && isequal(a.static, b.static) &&
            isequal(a.pos, b.pos) && isequal(a.stopped, b.stopped))
end

function Base.hash(a::DomainState, h::UInt)
    h = hash(a.pos, h)
    h = hash(a.trailing, h)
    h = hash(a.oncoming, h)
    h = hash(a.dynamic, h)
    h = hash(a.stopped, h)
    return h
end

struct DomainAction
    value::Int
end

function ==(a::DomainAction, b::DomainAction)
    return isequal(a.value, b.value)
end

function Base.hash(a::DomainAction, h::Uint)
    return hash(a.value, h)
end

struct DomainSSP
    S::Vector{DomainState}
    A::Vector{DomainAction}
    T::Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}}
    C::function
   s₀::DomainState
    G::Set{DomainState}
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

function index(state::DomainState, M::DomainSSP)
    return M.SIndex[state]
end
function index(action::DomainAction, M::DomainSSP)
    return M.AIndex[action]
end

function generate_states()
    S = Vector{DomainState}()
    G = Set{DomainState}()

    for pos in 1:4
        for t in [false, true]
            for o in [false, true]
                for d in [false, true]
                    for s in [false, true]
                        state = DomainState(pos, t, o, d, s)
                        push!(S, state)
                        if pos == 5
                            push!(G, state)
                        end
                    end
                end
            end
        end
    end

    return S, G
end

function terminal(state::DomainState)
    return state.pos == 4
end

function generate_actions()
    A = Vector{DomainAction}()
    push!(A, DomainAction("stop"))
    push!(A, DomainAction("edge"))
    push!(A, DomainAction("go"))

    return A
end

function stop_distribution(S::Vector{DomainState}
                           s::DomainState
end

function edge_distribution(S::Vector{DomainState}
                           s::DomainState)

end

function go_distribution(S::Vector{DomainState}
                           s::DomainState)

end

function generate_transitions()
    S, A, T = M.S, M.A, M.T

    for (s, state) in enumerate(S)
        T[s] = Dict{Int, Vector{Pair{Int, Float64}}}()
        if terminal(M, state)
            for (a, action) in enumerate(A)
                T[s][a] = [(s, 1.0)]
            end
        end

        for (a, action) in enumerate(A)
            if action.value == "stop"
                T[s][a] = stop_distribution(S, state)
            elseif action.value == "edge"
                T[s][a] = edge_distribution(S, state)
            else
                T[s][a] = go_distribution(S, state)
            end

        end


    end
end

function stop_distribution()

end

function edge_distribution()

end

function go_distribution()

end

function generate_costs()

end
