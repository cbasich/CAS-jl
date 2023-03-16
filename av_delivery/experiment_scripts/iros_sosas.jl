include("../scripts/utils.jl")

function simulate_sosas(M::SOSAS, num_runs)
    @time L = solve_model(M)
    S, A, C = M.S, M.A, M.C

    total_costs = Vector{Float64}()
    domain_costs = Vector{Float64}()
    human_costs = Vector{Float64}()

    for i=1:num_runs
        state = M.s₀
        episode_cost = 0.
        domain_cost = 0.
        human_cost = 0.
        while !terminal(M, state) && episode_cost < 1000.0
            s = M.SIndex[state]
            a = solve(L, M, s)[1]
            # a = rand(1:2)
            action = A[a]
            # println("Operator $action taking action $action in state $state.")
            # println(episode_cost)
            # episode_cost += generate_costs(M, M.L1, M.L2, s, a)
            domain_cost += generate_costs(M, M.L1, M.L2, s, a)
            # println(generate_costs(M, M.L1, M.L2, s, a))
            human_cost += autonomy_cost(state)
            # println(autonomy_cost(state))
            if action.operator == 2
                human_cost += 1
            end
            episode_cost = (domain_cost + human_cost)

            state′ = generate_successor(M, state, action)
            # if state′.σ == '⊘'
            #     block_transition!(H, state, action)
            #     delete!(L.solved, s)
            # end
            state = state′
        end

        push!(total_costs, episode_cost)
        push!(domain_costs, domain_cost)
        push!(human_costs, human_cost)
    end

    return mean(total_costs), std(total_costs), mean(domain_costs), std(domain_costs), mean(human_costs), std(human_costs)
end

function run_sosas()
    AV = build_av()
    L1 = solve_model(AV)
    K = build_model()
    H = build_cas(K, [0], ['⊘', '⊕', '∅'])
    L2 = solve_model(H)
    M = build_sosas(AV, L1, H, L2)

    saved_results = []
    w = WorldState(2, "day", "sunny")
    tasks = [v for (k,v) in fixed_routes]
    episode = 1
    for (init, goal) in tasks #[(12, 10), (16,4)]#[(2, 72), (72, 16), (12, 2)] #tasks
        set_route(M, init, goal, w)
        println(episode, "   |   Task: $init --> $goal")
        results = simulate_sosas(M, 1000)
        push!(saved_results, results)
        save_object(joinpath(abspath(@__DIR__), "SOSAS_results.jld2"), results)
        println(results)
        episode += 1
    end
end

run_sosas()
