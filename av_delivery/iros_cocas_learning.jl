include("utils.jl")
include("../LAOStarSolver.jl")
include("../LRTDPsolver.jl")
include("co_competence_aware_system.jl")

# using Infiltrator
function simulate(𝒮, ℒ, visited, num_runs)
    S, A, C = 𝒮.S, 𝒮.A, 𝒮.C
    T_base = deepcopy(𝒮.T)
    L_base = deepcopy(ℒ)
    costs = Vector{Float64}()
    signal_count, actions_taken, actions_at_comp, queries = 0, 0, 0, 0
    operator_state = generate_random_operator_state()
    for i = 1:num_runs
        L = L_base
        COCAS = L.M
        state = COCAS.s₀
        sh = operator_state
        episode_cost = 0.

        while !terminal(COCAS, state) && episode_cost < 1000.
            s = COCAS.SIndex[state]
            push!(visited, COCAS.𝒮.D.SIndex[state.state])
            a = solve(L, COCAS, s)[1]
            action = A[a]
            actions_taken += 1
            actions_at_comp += (action.l == competence(state.state, action.action))
            println("$i   |   Taking action $action in state $state.")
            σ = '⊕'
            if action.l == 0 || action.l == 1
                queries += 1
                o = state.sh[3]
                σ = generate_feedback(state, action, get_consistency(sh))
                if i == num_runs
                    y = (σ == '⊕' || σ == '∅') ? 1 : 0
                    d = hcat(get_state_features(state.state), o, y)
                    if typeof(state.state) == NodeState
                        𝒮.𝒮.F.D[o]["node"][string(action.action.value)][action.l] = record_data!(
                            d, 𝒮.𝒮.F.D[o]["node"][string(action.action.value)][action.l])
                    else
                        𝒮.𝒮.F.D[o]["edge"][string(action.action.value)][action.l] = record_data!(
                            d, 𝒮.𝒮.F.D[o]["edge"][string(action.action.value)][action.l])
                    end
                end
            end

            episode_cost += C[s][a]

            if σ == '⊖' || σ == '⊘'
                if σ == '⊖'
                    block_transition!(COCAS, state, action)
                    empty!(L.solved)
                    L.V[s] *= 0.0
                    # @infiltrate
                end
                TH = human_state_transition(sh, state.state, action.action, action.l)
                sh = sample(first.(TH), aweights(last.(TH)))
                w = state.state.w
                if w.active_avs == 4
                    w = WorldState(1, w.time, w.weather)
                else
                    w = WorldState(w.active_avs+1, w.time, w.weather)
                end
                if typeof(state.state) == NodeState
                    dstate′ = NodeState(state.state.id, state.state.p,
                        state.state.o, state.state.v, state.state.θ, w)
                else
                    dstate′ = EdgeState(state.state.u, state.state.v,
                        state.state.θ, state.state.o, state.state.l,
                        state.state.r, w)
                end
                state = COCASstate(sh, dstate′, σ)
                # delete!(L.solved, s)
                # COCAS.s₀ = state
                # L = solve_model(COCAS)
                continue
            else
                state = generate_successor(COCAS, s, a, σ)
                # state = generate_successor(COCAS.𝒮.D, state, action, σ)
            end
        end

        push!(costs, episode_cost)
        COCAS.T = T_base
        L = L_base
        # empty!(L.solved)
        # L.V *= 0.0
    end

    return mean(costs), std(costs), (actions_at_comp / actions_taken), (queries / num_runs)
end

function run_cocas()
    #tasks = load_object(joinpath(abspath(@__DIR__), "tasks.jld2"))
    #world_states = load_object(joinpath(abspath(@__DIR__), "world_states.jld2"))
    init_data()
    los_all_full, los_visited_full = Vector{Float64}(), Vector{Float64}()
    los_reach_full, los_reach_pol = Vector{Float64}(), Vector{Float64}()
    costs = Vector{Float64}()
    stds = Vector{Float64}()
    operative_LOs = Vector{Float64}()
    total_average_queries_to_human = [0.0]
    visited = Set{Int}()

    results = []
    D = build_model()
    C = build_cocas(D, [0,1,2], ['⊕', '⊖', '⊘', '∅'])
    for episode=1:200
        init, goal = (12, 10) #tasks[episode]
        w = WorldState(2, "day", "sunny") #world_states[episode]
        println(episode) #, "   |   Task: $init --> $goal")

        println("Building models...")
        set_route(D, C, init, goal, w)
        try
            generate_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)
        catch
            init_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)
        end

        println("Solving...")
        @time L = solve_model(C)

        println("Simulating...")
        c, std, operative_LO, average_queries = simulate(C, L, visited, 10)
        push!(costs, c), push!(stds, std), push!(operative_LOs, operative_LO)
        push!(total_average_queries_to_human, average_queries + last(total_average_queries_to_human))

        if episode%5 == 0
            println("Updating feedback profile...")
            update_feedback_profile!(C)
            println("Updating autonomy profile...")
            update_autonomy_profile!(C, L)
            println("Saving...")
            save_data(C.𝒮.F.D)
            save_object(joinpath(abspath(@__DIR__), "COCAS_params.jld2"), (C.𝒮.A.κ, C.𝒮.F.λ))
        end

        if episode == 1 || episode%5 == 0
            lo_all_full, lo_visited_full = compute_level_optimality(C, visited)
            lo_reach_full, lo_reach_pol = compute_reachable_level_optimality(C, L)
            push!(los_all_full, lo_all_full)
            push!(los_visited_full, lo_visited_full)
            push!(los_reach_full, lo_reach_full)
            push!(los_reach_pol, lo_reach_pol)
            # lo_all_opt = compute_level_optimality(C, L)
            # push!(los, lo)
            println("LO: $lo_all_full | $lo_visited_full | $lo_reach_full
                                      | $lo_reach_pol | $operative_LO")
            results = [costs, stds, los_all_full, los_visited_full, los_reach_full,
                       los_reach_pol, operative_LOs, total_average_queries_to_human[2:end]]
            save_object(joinpath(abspath(@__DIR__), "COCAS_results.jld2"), results)
        end
    end
end

run_cocas()
