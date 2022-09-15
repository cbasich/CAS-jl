include("utils.jl")
include("../LAOStarSolver.jl")
include("../LRTDPsolver.jl")
include("competence_aware_system.jl")

function simulate(CAS, L, num_runs)
    S, A, C = CAS.S, CAS.A, CAS.C
    T_base = deepcopy(CAS.T)
    costs = Vector{Float64}()
    operator_state = generate_random_operator_state()

    for i = 1:num_runs
        state = CAS.s₀
        sh = operator_state
        episode_cost = 0.

        while !terminal(CAS, state) && episode_cost < 1000.
            s = CAS.SIndex[state]
            a = solve(L, CAS, s)[1]
            action = A[a]
            # println("$i   |   Taking action $action in state $state with operator $sh.")
            σ = '⊕'
            if action.l == 0
                σ = generate_feedback(state, action, sh, get_consistency(sh))
            end

            episode_cost += C[s][a]
            TH = human_state_transition(sh, state.state, action.action, action.l)
            sh = sample(first.(TH), aweights(last.(TH)))
            state = generate_successor(CAS.𝒮.D, state, action, σ)
        end

        push!(costs, episode_cost)
        CAS.T = T_base
    end

    return mean(costs), std(costs)
end

function run_cas()
    tasks = [v for (k,v) in fixed_routes]
    costs = Vector{Float64}()
    stds = Vector{Float64}()

    results = []
    D = build_model()
    C = build_cas(D, [0,1,2], ['⊕', '⊖', '⊘', '∅'])

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
        save_object(joinpath(abspath(@__DIR__), "CAS_results.jld2"), results)

        episode += 1
    end
end

run_cas()
