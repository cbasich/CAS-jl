include("utils.jl")
include("../LAOStarSolver.jl")
include("../LRTDPsolver.jl")
include("av_only.jl")
include("human_operator.jl")
include("so_sas.jl")

function simulate_human_only(H, num_runs)
    @time L = solve_model(H)

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

function run_human_only()
    D = build_model()
    H = build_cas(D, [0], ['⊘', '∅'])

    tasks = [v for (k,v) in fixed_routes]
    w = WorldState(2, "day", "sunny")
    episode = 1
    for (init, goal) in tasks
        set_route(D, H, init, goal, w)
        generate_transitions!(H.𝒮.D, H.𝒮.A, H.𝒮.F, H, H.S, H.A, H.G)

        println(episode, "   |   Task: $init --> $goal")
        results = simulate_human_only(H, 1000)
        println(results)

        episode += 1
    end
end

run_human_only()
