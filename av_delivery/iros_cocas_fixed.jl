include("utils.jl")
include("../LAOStarSolver.jl")
include("../LRTDPsolver.jl")
include("co_competence_aware_system.jl")

function simulate(COCAS, L, num_runs)
    S, A, C = COCAS.S, COCAS.A, COCAS.C
    T_base = deepcopy(COCAS.T)
    costs = Vector{Float64}()
    operator_state = generate_random_operator_state()

    for i = 1:num_runs
        state = COCAS.s₀
        sh = operator_state
        episode_cost = 0.

        while !terminal(COCAS, state) && episode_cost < 1000.
            s = COCAS.SIndex[state]
            a = solve(L, COCAS, s)[1]
            action = A[a]
            # println("$i   |   Taking action $action in state $state with operator $sh.")
            σ = '⊕'
            if action.l == 0
                σ = generate_feedback(state, action, sh, get_consistency(sh))
            end

            episode_cost += C[s][a]
            state = generate_successor(COCAS.𝒮.D, state, action, σ)
        end

        push!(costs, episode_cost)
        COCAS.T = T_base
    end

    return mean(costs), std(costs)
end

function run_cocas()
    tasks = [v for (k,v) in fixed_routes]
    costs = Vector{Float64}()
    stds = Vector{Float64}()

    results = []
    D = build_model()
    C = build_cocas(D, [0,1,2], ['⊕', '⊖', '⊘', '∅'])

    w = WorldState(2, "day", "sunny")
    episode = 1
    for (init, goal) in tasks
        set_route(D, C, init, goal, w)
        generate_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)

        println(episode, "   |   Task: $init --> $goal")
        @time L = solve_model(C)
        c, std = simulate(C, L, 10)
        println(c, "  |  ", std)
        push!(costs, c), push!(stds, std)

        results = [costs, stds]
        save_object(joinpath(abspath(@__DIR__), "COCAS_results.jld2"), results)

        episode += 1
    end
end

run_cocas()
