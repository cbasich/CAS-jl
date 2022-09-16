include("utils.jl")
include("../LAOStarSolver.jl")
include("../LRTDPsolver.jl")
include("co_competence_aware_system.jl")

function simulate(COCAS, L, num_runs)
    S, A, C = COCAS.S, COCAS.A, COCAS.C
    T_base = deepcopy(COCAS.T)
    total_costs = Vector{Float64}()
    domain_costs = Vector{Float64}()
    human_costs = Vector{Float64}()

    for i = 1:num_runs
        state = COCAS.s₀
        episode_cost = 0.
        domain_cost = 0.
        h_cost = 0.
        while !terminal(COCAS, state) && episode_cost < 1000.
            s = COCAS.SIndex[state]
            a = solve(L, COCAS, s)[1]
            action = A[a]
            # println("$i   |   Taking action $action in state $state with operator $sh."

            cost = C[s][a]
            cost2 = autonomy_cost(state) + human_cost(action)

            h_cost += cost2
            domain_cost += (cost - cost2)
            episode_cost += cost

            σ = '⊕'
            if action.l == 0
                σ = generate_feedback(state, action, get_consistency(state.sh))
            end
            state = generate_successor(COCASs, s, a, σ)
        end

        push!(total_costs, episode_cost)
        push!(domain_costs, domain_cost)
        push!(human_costs, h_cost)
        COCAS.T = T_base
    end

    return mean(total_costs), std(total_costs), mean(domain_costs), std(domain_costs), mean(human_costs), std(human_costs)
end

function run_cocas()
    tasks = [v for (k,v) in fixed_routes]
    costs = Vector{Float64}()
    stds = Vector{Float64}()

    saved_results = []
    D = build_model()
    C = build_cocas(D, [0,1,2], ['⊕', '⊖', '⊘', '∅'])

    w = WorldState(2, "night", "snowy")
    episode = 1
    for (init, goal) in tasks
        set_route(D, C, init, goal, w)
        generate_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)

        println(episode, "   |   Task: $init --> $goal")
        @time L = solve_model(C)
        results = simulate(C, L, 1000)
        println(results)
        push!(saved_results, results)
        save_object(joinpath(abspath(@__DIR__), "COCAS_results.jld2"), saved_results)

        episode += 1
    end
end

run_cocas()
