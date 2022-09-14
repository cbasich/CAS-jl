include("utils.jl")
include("../LAOStarSolver.jl")
include("../LRTDPsolver.jl")

include("co_competence_aware_system.jl")
include("competence_aware_system.jl")
include("human_operator.jl")
include("av_only.jl")
include("so_sas.jl")


function generate_test_tasks()
    tasks = []
    for i=1:500
        push!(tasks, rand(fixed_routes)[2])
    end
    save_object(joinpath(abspath(@__DIR__), "tasks.jld2"), tasks)
    return tasks
end

function load_test_tasks()
    return load_object(joinpath(abspath(@__DIR__), "tasks.jld2"))
end

##
##
##FULL AUTONOMY ONLY

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
            println("taking action $action in state $state.")
            episode_cost += C[s][a]
            state = AV.S[generate_successor(AV, s, a)]

            if typeof(state) == EdgeState
                if state.o && state.l == 1
                    episode_cost = 1000.0
                    break
                end
            end

            if episode_cost > 1000.0
                break
            end
        end

        if terminal(AV, state)
            println("Reached the goal!\n\n")
        else
            println("Terminated in state $state.\n\n")
        end
        push!(costs, episode_cost)
    end

    return mean(costs), std(costs)
end

function run_full_autonomy(tasks)
    AV = build_model()

    episode = 0
    for (init, goal) in tasks
        if init == 12 || goal == 12 || init == 11 || goal == 11
            continue
        end
        w = generate_random_world_state()
        set_init!(AV, init, w)
        set_goals!(AV, [goal], w)
        generate_transitions!(AV, AV.graph)
        generate_costs!(AV)
        println(episode, "   |   Task: $init --> $goal")
        results = simulate(AV, 100)
        println(results)
    end
end

run_full_autonomy(load_test_tasks())

##
##
## HUMAN OPERATOR ONLY

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

function run_human_only(tasks)
    D = build_model()
    H = build_cas(D, [0], ['⊘', '∅'])

    episode = 1
    for (init, goal) in tasks
        w = generate_random_world_state()
        set_route(D, H, init, goal, w)
        generate_transitions!(H.𝒮.D, H.𝒮.A, H.𝒮.F, H, H.S, H.A, H.G)

        println(episode, "   |   Task: $init --> $goal")
        results = simulate_human_only(H, 1000)
        println(results)

        episode += 1
    end
end

run_human_only(load_test_tasks())

##
##
## SOSAS

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

function run_sosas(tasks)
    D = build_model()
    L1 = solve_model(D)
    H = build_cas(D, [0], ['⊘', '∅'])
    L2 = solve_model(H)
    SOSAS = build_sosas(D, L1, H, L2)

    episode = 1
    for (init, goal) in tasks
        w = generate_random_world_state()
        set_route(D, H, init, goal, w)
        generate_transitions!(H.𝒮.D, H.𝒮.A, H.𝒮.F, H, H.S, H.A, H.G)

        println(episode, "   |   Task: $init --> $goal")
        results = simulate_sosas(H, 1000)
        println(results)
    end
end

run_sosas(load_test_tasks())

##
##
## CAS

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

            println("$i   |   Taking action $action in state $state with operator $sh.")

            σ = '⊕'
            if action.l == 0 || action.l == 1
                σ = generate_feedback(state, action, sh, get_consistency(sh))
                if i == num_runs
                    y = (σ == '⊕' || σ == '∅') ? 1 : 0
                    d = hcat(get_state_features(state.state), action.l, y)
                    if typeof(state.state) == NodeState
                        CAS.𝒮.F.D["node"][string(action.action.value)] = record_data!(
                            d, CAS.𝒮.F.D["node"][string(action.action.value)])
                    else
                        CAS.𝒮.F.D["edge"][string(action.action.value)] = record_data!(
                            d, CAS.𝒮.F.D["edge"][string(action.action.value)])
                    end
                end
            end

            episode_cost += C[s][a]
            if σ == '⊖'
                block_transition!(CAS, state, action)
                state = CASstate(state.state, σ)
                delete!(L.solved, s)
                continue
            else
                TH = human_state_transition(sh, state.state, action.action, action.l)
                sh = sample(first.(TH), aweights(last.(TH)))
                state = generate_successor(CAS.𝒮.D, state, action, σ)
            end
        end

        push!(costs, episode_cost)
        CAS.T = T_base
    end

    return mean(costs), std(costs), (actions_at_comp / actions_taken)
end

function run_cas(tasks)
    println("Starting")
    init_data()
    los, los_r = Vector{Float64}(), Vector{Float64}()
    costs = Vector{Float64}()
    stds = Vector{Float64}()
    total_signals_received = 0

    results = []
    episode = 1
    D = build_model()
    C = build_cas(D, [0,1,2], ['⊕', '⊖', '⊘', '∅'])
    for (init, goal) in tasks
        w = generate_random_world_state()
        set_route(D, C, init, goal, w)
        generate_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)

        println(episode, "   |   Task: $init --> $goal")
        @time L = solve_model(C)
        c, std, percent_lo = simulate(C, L, 100)
        push!(costs, c), push!(stds, std)

        if episode%5 == 0
            update_feedback_profile!(C)
            update_autonomy_profile!(C, L)
            save_data(C.𝒮.F.D)
            results = [costs, stds, los, los_r]
            save_object(joinpath(abspath(@__DIR__), "CAS_results.jld2"), results)
        end

        if episode == 1 || episode%10 == 0
            lo, lo_r = compute_level_optimality(C, L)
            push!(los, lo), push!(los_r, lo_r)
            println("LO: $lo  | $lo_r")
        end

        episode += 1
    end
end

run_cas(load_test_tasks())
results = load_object(joinpath(abspath(@__DIR__), "CAS_results.jld2"))
