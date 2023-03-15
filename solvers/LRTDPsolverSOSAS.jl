using Base

mutable struct LRTDPsolverSOSAS
    M
    dead_end_cost::Float64
    max_trials::Int
    ϵ::Float64
    π::Dict{Int, Int}
    dont_label::Bool
    solved::Set{Int}
    H::Vector{Float64}
    V::Vector{Float64}
    Q::Vector{Float64}
end

function lookahead(solver, s, a)
    q, V, H, M = 0., solver.V, solver.H, solver.M
    if typeof(solver.M) == SOSAS
        T = get_transition(M, s, a)
    else
        T = solver.M.T[s][a]
    end
    for i = 1:length(T)
    # for (sp, p) in T
        if haskey(solver.π, T[i][1])
            q += T[i][2] * V[T[i][1]]
        # else
        #     q += T[i][2] * H[M.𝒮.D.SIndex[M.S[T[i][1]].state]]
            # println(T[i][2], "|", H[M.𝒮.D.SIndex[M.S[T[i][1]].state]], "|", T[i][1])
            # println(q)
        end
    end
    # return q + solver.M.C(solver.M, s, a)
    if typeof(solver.M) == SOSAS
        return q + generate_costs(solver.M, solver.M.L1, solver.M.L2, s, a)
    end
    return q + solver.M.C[s][a]
end

function backup(solver, s)
    for a = 1:length(solver.M.A)
        if !allowed(solver.M, s, a)
            solver.Q[a] = max(solver.dead_end_cost, lookahead(solver, s, a))
            # println(a)
        else
            solver.Q[a] = lookahead(solver, s, a)
        end
        # solver.Q[a] = lookahead(solver, s, a)
    end
    a = Base.argmin(solver.Q)
    return a, solver.Q[a]
end

function bellman_update(solver, s)
    a, q = backup(solver, s)
    residual = abs(solver.V[s] - q)
    solver.V[s] = q
    solver.π[s] = a
    # println(s, " | ", residual)
    return residual
end

function solve(solver::LRTDPsolverSOSAS, M, s)
    trials = 0
    # println("Solving...$s")
    while s ∉ solver.solved && trials < solver.max_trials
        # println(trials, "  |  $s")
        trial(solver, s)
        trials += 1
    end
    # println(trials)
    bellman_update(solver, s)
    return solver.π[s]
end

function generate_successor(T)
    r = rand()
    thresh = 0.
    for i=1:length(T)
        thresh += T[i][2]
        if r <= thresh
            return T[i][1]
        end
    end
    println(T)
    println("No successor found")
end

function trial(solver::LRTDPsolverSOSAS, s)
    total_cost = 0.
    visited = Set()
    M = solver.M
    # println(s∉ solver.solved)
    while s ∉ solver.solved
        # println(s)
        if total_cost > solver.dead_end_cost
            break
        end
        state = M.S[s]
        if terminal(M, state)
            # println("Terminating...")
            # total_cost += autonomy_cost(state)
            break
        end
        # println(s)

        push!(visited, s)
        bellman_update(solver, s)

        a = solver.π[s]
        # println(s, "    |    ", a)
        if typeof(solver.M) == SOSAS
            total_cost += generate_costs(solver.M, solver.M.L1, solver.M.L2, s, a)
            s = generate_successor(get_transition(solver.M, s, a))
            # println(total_cost)
        else
            total_cost += M.C[s][a]
            # println(a)
            s = generate_successor(M.T[s][a])
        end
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

function check_solved(solver::LRTDPsolverSOSAS, s)
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

        # println(residual)
        if residual > solver.ϵ
            rv = false
        end

        if typeof(solver.M) == SOSAS
            T = get_transition(solver.M, s, a)
        else
            T = solver.M.T[s][a]
        end

        for (sp, p) in T #solver.M.T[s][a]
            if sp ∉ solver.solved && sp ∉ _open && sp ∉ _closed
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
