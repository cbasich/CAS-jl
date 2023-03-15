using Base

mutable struct ValueIterationSolver
    eps::Float64
    cost_minimizing::Bool
    π::Dict{Integer, Integer}
    V::Vector{Float64}
    Qs::Vector{Float64}
end
# function ValueIterationSolver(eps, cost_minimizing)
#     return ValueIterationSolver(eps, cost_minimizing)
# end

function lookahead(𝒱::ValueIterationSolver, M, s::Integer, a::Integer)
    S, T, C, V = M.S, M.T[s][a], M.C, 𝒱.V
    # return C[s][a] + 0.99 * sum(T[s][a][s′] * V[s′] for s′=1:length(S))
    q = 0.
    for i=1:length(T)
        q += T[i][2] * V[T[i][1]]
    end
    return C[s][a] + q
end

function backup(𝒱::ValueIterationSolver, M, s::Integer)
    for a = 1:length(M.A)
        𝒱.Qs[a] = lookahead(𝒱, M, s, a)
    end
    a = Base.argmin(𝒱.Qs)
    q = 𝒱.Qs[a]
    return a, q
end

function solve(𝒱::ValueIterationSolver, M)
    𝒱.V = Vector{Float64}(undef, length(M.S))
    𝒱.Qs = Vector{Float64}(undef, length(M.A))

    last_res = 0.
    while true
        residual = 0.
        res_s, res_a = 1, 1
        for s = 1:length(M.S)
            if M.S[s].state.w != M.s₀.state.w || (M.S[s].sh[1] == 1 && M.S[s].sh[3] == 2)
                continue
            end
            a, q = backup(𝒱, M, s)
            if abs(𝒱.V[s] - q) > residual
                residual = abs(𝒱.V[s] - q)
                res_s, res_a = s, a
            end
            # residual = max(residual, abs(𝒱.V[s] - q))
            𝒱.V[s] = q
            𝒱.π[s] = a
        end
        println(residual)
        println(res_s, "|", res_a)
        if residual < 𝒱.eps
            break
        end
        # println(last_res)
        # if last_res != 0.
        #     if last_res - residual < 0.001
        #         break
        #     end
        # end
        # last_res = residual
    end
end
