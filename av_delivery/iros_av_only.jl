include("utils.jl")
include("../LAOStarSolver.jl")
include("../LRTDPsolver.jl")
include("av_only.jl")

function simulate(AV, num_runs)
    @time L = solve_model(AV)
    S, A, C = AV.S, AV.A, AV.C
    costs = Vector{Float64}()

    for i=1:num_runs
        state = AV.s₀
        episode_cost = 0.0

        while !terminal(AV, state)
            s = AV.SIndex[state]
            a = solve(L, AV, s)[1]
            action = A[a]
            # println("taking action $action in state $state.")
            episode_cost += C[s][a]
            state = AV.S[generate_successor(AV, s, a)]

            if typeof(state) == EdgeState
                if state.o && state.l == 1
                    episode_cost = 10000.0
                    break
                end
            end

            if episode_cost > 10000.0
                break
            end
        end

        if terminal(AV, state)
            # println("Reached the goal!\n\n")
        else
            # println("Terminated in state $state.\n\n")
        end
        push!(costs, episode_cost)
    end

    return mean(costs), std(costs)
end

function run_full_autonomy()
    AV = build_av()
    tasks = [v for (k,v) in fixed_routes]
    episode = 0
    saved_results = []
    for (init, goal) in tasks
        w = generate_random_world_state()
        set_init!(AV, init, w)
        set_goals!(AV, [goal], w)
        generate_transitions!(AV, AV.graph)
        generate_costs!(AV)
        println(episode, "   |   Task: $init --> $goal")
        results = simulate(AV, 100)
        println(results)
        push!(saved_results, results)
        save_object(joinpath(abspath(@__DIR__), "AVONLY_results.jld2"), saved_results)
        episode += 1
    end
end

run_full_autonomy()
