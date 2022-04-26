using Base

mutable struct LAOStarSolver
    max_iter::Integer
    dead_end_cost::Float64
    ω::Float64
    ϵ::Float64
    π::Dict{Integer,Integer}
    V::Vector{Float64}
    G::Vector{Float64}
    H::Vector{Float64}
    Qs::Vector{Float64}
    solved::Vector{Bool}
end

function lookahead(ℒ::LAOStarSolver,
                   M,
                   s::Integer,
                   a::Integer)
    S, A, T, C, V = M.S, M.A, M.T[s][a], M.C, ℒ.V

    q = 0.
    for (s′, p) in T
        if haskey(ℒ.π, s′)
            q += p * V[s′]
        end
    end
    return q + C(M,s,a)
end

function backup(ℒ::LAOStarSolver,
                M,
                s::Integer)
    for a = 1:length(M.A)
        # ℒ.Qs[a] = lookahead(ℒ, M, s, a)
        if !allowed(M, s, a)
            ℒ.Qs[a] = 1000.0
        else
            ℒ.Qs[a] = lookahead(ℒ, M, s, a)
        end
    end
    a = Base.argmin(ℒ.Qs)
    return a, ℒ.Qs[a]
end

function bellman_update(ℒ::LAOStarSolver,
                        M,
                        s::Integer)

    a, q = backup(ℒ, M, s)
    residual = abs(ℒ.V[s] - q)
    ℒ.V[s] = q
    ℒ.π[s] = a
    return residual
end

function expand(ℒ::LAOStarSolver,
                M,
                s::Integer,
          visited::Set{Integer})
    # println(M.S[s])
    if s ∈ visited
        return 0
    end
    push!(visited, s)
    if terminal(M, M.S[s])
        return 0
    end
    count = 0
    if s ∉ keys(ℒ.π)
        bellman_update(ℒ, M, s)
        return 1
    else
        a = ℒ.π[s]
        # println(M.A[a])
        for (s′, p) in M.T[s][a]
            count += expand(ℒ, M, s′, visited)
        end
    end
    return count
end

function test_convergence(ℒ::LAOStarSolver,
                          M,
                          s::Integer,
                    visited::Set{Integer})
    error = 0.0
    if terminal(M, M.S[s])
        return 0.0
    end
    # println(visited)
    if s ∈ visited
        return 0.0
    end
    push!(visited, s)
    a = -1
    if s ∈ keys(ℒ.π)
        a = ℒ.π[s]
        for (s′, p ) in M.T[s][a]
            error = max(error, test_convergence(ℒ, M, s′, visited))
        end
    else
        return ℒ.dead_end_cost + 1
    end

    error = max(error, bellman_update(ℒ, M, s))
    if (a == -1 && !haskey(ℒ.π, s)) || (haskey(ℒ.π, s) && a == ℒ.π[s])
        return error
    end
    return ℒ.dead_end_cost + 1
end

function solve(ℒ::LAOStarSolver,
               M,
               s::Integer)

    if ℒ.solved[s]
        return ℒ.π[s]
    end
    expanded = 0
    iter = 0
    total_expanded = 0
    error = ℒ.dead_end_cost

    visited = Set{Integer}()
    while true
        while true
            empty!(visited)
            num_expanded = expand(ℒ, M, s, visited)
            total_expanded += num_expanded
            # println(num_expanded, "               ", total_expanded)
            if num_expanded == 0
                break
            end
        end
        while true
            empty!(visited)
            error = test_convergence(ℒ, M, s, visited)
            # println("State $(M.S[s])    |    $error")
            if error > ℒ.dead_end_cost
                break
            end
            if error < ℒ.ϵ
                if !haskey(ℒ.π, s)
                    println(M.S[s])
                end
                # println("$s:  Total nodes expanded: $total_expanded")
                ℒ.solved[s] = true
                return ℒ.π[s], total_expanded
            end
        end
        iter += 1
    end
    println("Total nodes expanded: $total_expanded")
    ℒ.solved[s] = true
    return ℒ.π[s], total_expanded
end
