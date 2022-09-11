using Base

mutable struct LRTDPsolver
    M
    dead_end_cost::Float64
    max_trials::Int
    ϵ::Float64
    π::Dict{Int, Int}
    dont_label::Bool
    solved::Set{Int}
    V::Vector{Float64}
    Q::Vector{Float64}
end

function lookahead(solver, s, a)
    q = 0.
    for (sp, p) in solver.M.T[s][a]
        if haskey(solver.π, sp)
            q += p * solver.V[sp]
        end
    end
    return q + solver.M.C(solver.M, s, a)
end

function backup(solver, s)
    for a = 1:length(solver.M.A)
        if !allowed(solver.M, s, a)
            solver.Q[a] = solver.dead_end_cost
        else
            solver.Q[a] = lookahead(solver, s, a)
        end
    end
    a = Base.argmin(solver.Q)
    return a, solver.Q[a]
end

function bellman_update(solver, s)
    a, q = backup(solver, s)
    residual = abs(solver.V[s] - q)
    solver.V[s] = q
    solver.π[s] = a
    return residual
end

function solve(solver, M, s)
    trials = 0
    while s ∉ solver.solved && trials < solver.max_trials
        trial(solver, s)
        trials += 1
    end
    return solver.π[s]
end

function generate_successor(T)
    r = rand()
    thresh = 0.
    for (sp, p) in T
        thresh += p
        if r <= thresh
            return sp
        end
    end
end

function trial(solver, s)
    total_cost = 0.
    visited = Set()

    while s ∉ solver.solved
        # println(s)
        if total_cost > solver.dead_end_cost
            break
        end
        state = solver.M.S[s]
        if terminal(solver.M, state)
            total_cost += autonomy_cost(state)
            break
        end
        # println(s)

        push!(visited, s)
        bellman_update(solver, s)

        a = solver.π[s]
        total_cost += solver.M.C(solver.M, s, a)

        s = generate_successor(solver.M.T[s][a])
    end

    if solver.dont_label
        return
    end

    while !isempty(visited)
        s = pop!(visited)
        if !check_solved(solver, s)
            return
        end
    end
end

function check_solved(solver, s)
    rv = true

    _open = Set()
    _closed = Set()

    if s ∉ solver.solved
        push!(_open, s)
    end

    while !isempty(_open)
        s = pop!(_open)
        if solver.M.S[s] ∈ solver.M.G
            continue
        end

        residual = bellman_update(solver, s)
        a = solver.π[s]
        push!(_closed, s)

        if residual > solver.ϵ
            rv = false
        end

        for (sp, p) in solver.M.T[s][a]
            if sp ∉ solver.solved && sp ∉ union(_open, _closed)
                push!(_open, sp)
            end
        end
    end

    if rv
        for sp in _closed
            push!(solver.solved, sp)
        end
    else
        while !isempty(_closed)
            s = pop!(_closed)
            bellman_update(solver, s)
        end
    end

    return rv
end
