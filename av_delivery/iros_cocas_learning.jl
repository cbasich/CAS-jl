include("utils.jl")
include("../LAOStarSolver.jl")
include("../LRTDPsolver.jl")
include("co_competence_aware_system.jl")

function simulate(CAS, L, num_runs)
    S, A, C = CAS.S, CAS.A, CAS.C
    T_base = deepcopy(CAS.T)
    costs = Vector{Float64}()
    signal_count, actions_taken, actions_at_comp = 0, 0, 0

    operator_state = generate_random_operator_state()
    for i = 1:num_runs
        state = CAS.s₀
        sh = operator_state
        episode_cost = 0.

        while !terminal(CAS, state) && episode_cost < 1000.
            s = CAS.SIndex[state]
            a = solve(L, CAS, s)[1]
            action = A[a]
            actions_taken += 1
            actions_at_comp += (action.l == competence(state.state, action.action))
            # println("$i   |   Taking action $action in state $state.")
            σ = '⊕'
            if action.l == 0 || action.l == 1
                o = state.sh[3]
                σ = generate_feedback(state, action, get_consistency(sh))
                if i == num_runs
                    y = (σ == '⊕' || σ == '∅') ? 1 : 0
                    d = hcat(get_state_features(state.state), o, action.l, y)
                    if typeof(state.state) == NodeState
                        CAS.𝒮.F.D[o]["node"][string(action.action.value)] = record_data!(
                            d, CAS.𝒮.F.D[o]["node"][string(action.action.value)])
                    else
                        CAS.𝒮.F.D[o]["edge"][string(action.action.value)] = record_data!(
                            d, CAS.𝒮.F.D[o]["edge"][string(action.action.value)])
                    end
                end
            end

            episode_cost += C[s][a]

            if σ == '⊖'
                block_transition!(CAS, state, action)
                TH = human_state_transition(sh, state.state, action.action, action.l)
                sh = sample(first.(TH), aweights(last.(TH)))
                state = COCASstate(sh, state.state, σ)
                delete!(L.solved, s)
                L.V[s] = 0.0
                # L = solve_model(C)
                continue
            elseif σ == '⊘'
                TH = human_state_transition(sh, state.state, action.action, action.l)
                sh = sample(first.(TH), aweights(last.(TH)))
                state = COCASstate(sh, state.state, σ)
            else
                # state = generate_successor(CAS, s, a, σ)
                state = generate_successor(CAS.𝒮.D, state, action, σ)
            end
        end

        push!(costs, episode_cost)
        CAS.T = T_base
    end

    return mean(costs), std(costs), (actions_at_comp / actions_taken)
end

function run_cas()
    tasks = load_object(joinpath(abspath(@__DIR__), "tasks.jld2"))
    init_data()
    los, los_r = Vector{Float64}(), Vector{Float64}()
    costs = Vector{Float64}()
    stds = Vector{Float64}()
    total_signals_received = 0

    results = []
    D = build_model()
    C = build_cocas(D, [0,1,2], ['⊕', '⊖', '⊘', '∅'])
    for episode=1:500
        init, goal = tasks[episode]
        println(episode, "   |   Task: $init --> $goal")

        println("Building models...")
        w = generate_random_world_state()
        set_route(D, C, init, goal, w)
        generate_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)


        println("Solving...")
        @time L = solve_model(C)
        println("Simulating...")
        c, std, percent_lo = simulate(C, L, 10)
        push!(costs, c), push!(stds, std)

        if episode%5 == 0
            println("Updating feedback profile...")
            update_feedback_profile!(C)
            println("Updating autonomy profile...")
            update_autonomy_profile!(C, L)
            println("Saving...")
            save_data(C.𝒮.F.D)
            save_object(joinpath(abspath(@__DIR__), "COCAS_params.jld2"), (C.𝒮.A.κ, C.𝒮.F.λ))
        end

        if episode == 1 || episode%10 == 0
            lo, lo_r = compute_level_optimality(C, L)
            push!(los, lo), push!(los_r, lo_r)
            println("LO: $lo  | $lo_r")
            results = [costs, stds, los, los_r]
            save_object(joinpath(abspath(@__DIR__), "COCAS_results.jld2"), results)
        end
    end
end

run_cas()
