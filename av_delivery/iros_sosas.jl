include("utils.jl")
include("../LAOStarSolver.jl")
include("../LRTDPsolver.jl")
include("human_operator.jl")
include("av_only.jl")
include("so_sas.jl")


function simulate_sosas(SOSAS, num_runs)
    @time L = solve_model(SOSAS)

    S, A, C = H.S, H.A, H.C
    T_base = deepcopy(H.T)

    costs = Vector{Float64}()
    for i=1:num_runs
        state = H.s₀
        episode_cost = 0.0

        while true
            s = H.SIndex[state]
            a = solve(L, H, s)[1]
            action = A[a]
            # println("taking action $action in state $state.")
            episode_cost += C[s][a]
            state′ = generate_successor(H, state, action)
            # if state′.σ == '⊘'
            #     block_transition!(H, state, action)
            #     delete!(L.solved, s)
            # end
            state = state′

            if terminal(H, state)
                break
            end
        end

        push!(costs, episode_cost)
        H.T = T_base
    end

    return mean(costs), std(costs)
end

function run_sosas()
    D = build_model()
    L1 = solve_model(D)
    H = build_cas(D, [0], ['⊘', '∅'])
    L2 = solve_model(H)
    SOSAS = build_sosas(D, L1, H, L2)

    w = WorldState(2, "day", "night")
    tasks = [v for (k,v) in fixed_routes]
    episode = 1
    for (init, goal) in tasks
        set_route(D, H, init, goal, w)
        generate_transitions!(H.𝒮.D, H.𝒮.A, H.𝒮.F, H, H.S, H.A, H.G)

        println(episode, "   |   Task: $init --> $goal")
        results = simulate_sosas(H, 1000)
        println(results)
        episode += 1
    end
end

run_sosas()
