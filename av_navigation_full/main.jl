using Profile,ProfileView
using Logging
using Infiltrator
include("../utils.jl")
include("competence_aware_system.jl")
include("../LRTDPsolver.jl")

function simulate(M::DomainSSP, C::CASSP, L, m)
    S, A = M.S, M.A
    c = Vector{Float64}()
    signal_count = 0
    actions_taken = 0
    actions_at_competence = 0
    expected_cost = L.V[M.SIndex[M.s₀]]
    # println("Expected cost to goal: $(ℒ.V[index(state, S)])")
    for i ∈ 1:m
        state = M.s₀
        episode_cost = 0.0
        while true
            s = M.SIndex[state]
            # println(state, "     ", s)
            a = solve(L, M, s)[1]
            # a = L.π[s]
            action = A[a]
            actions_taken += 1
            actions_at_competence += (2 == competence(state, action))
            # println("$i   |   Taking action $action in state $state.")
            σ = generate_feedback(CASstate(state, '∅'), CASaction(action, 2))
            episode_cost += C.C(C, C.SIndex[CASstate(state, σ)], C.AIndex[CASaction(action, 2)])

            if σ == '⊘'
                if i == m
                    signal_count += 1
                end
                state = M.S[M.T[s][a][1][1]]
            else
                state = M.S[generate_successor(M, s, a)]
            end
            # println(σ, "     | succ state |      ", state)
            if terminal(M, state)
                break
            end
        end

        push!(c, episode_cost)
    end
    println("Total cumulative reward: $(round(mean(c);digits=4)) ⨦ $(std(c))")
    return mean(c), std(c), signal_count, (actions_at_competence / actions_taken), (abs(mean(c) - expected_cost)/expected_cost)
end

function simulate(M::CASSP, L, m)
    override_rate_records_per_ep = Dict{DomainState, Array{Int}}()
    S, A, C = M.S, M.A, M.C
    T_base = deepcopy(M.T)
    c = Vector{Float64}()
    signal_count = 0
    actions_taken = 0
    actions_at_competence = 0
    expected_cost = L.V[M.SIndex[M.s₀]]
    # println("Expected cost to goal: $(ℒ.V[index(state, S)])")
    for i ∈ 1:m
        state = M.s₀
        episode_cost = 0.0
        while true
            s = M.SIndex[state]
            if !haskey(override_rate_records_per_ep, state.state)
                override_rate_records_per_ep[state.state] = [1 0]
            else
                override_rate_records_per_ep[state.state][1] += 1
            end
            # println(state, "     ", s)
            a = solve(L, M, s)[1]
            # a = L.π[s]
            action = A[a]
            if action.action.value == '⤉' && typeof(state) == NodeState
                println("WAITING @ state:    $state")
            end
            if action.action.value == '→' && M.𝒮.W.right_occupied
                M.𝒮.W.right_occupied = false
                M.𝒮.W.left_occupied = true
            end
            if action.action.value == '←' && M.𝒮.W.left_occupied
                M.𝒮.W.right_occupied = true
                M.𝒮.W.left_occupied = false
            end

            actions_taken += 1
            actions_at_competence += (action.l == competence(state.state, M.𝒮.W, action.action))
            # println("$i   |   Taking action $action in state $state.")
            if action.l == 0 || action.l == 3
                σ = '∅'
            elseif action.l == 1
                σ = generate_feedback(state, M.𝒮.W, action)
                if i == m
                    y = (σ == '⊕') ? 1 : 0
                    d = hcat(get_state_features(M, state.state), 1, y)
                    d_full = hcat(get_full_state_features(M, state.state), 1, y)
                    if typeof(state.state) == NodeState
                        # record_data(d,joinpath(abspath(@__DIR__), "data", "node_$(action.action.value).csv"))
                        M.𝒮.F.D["node"][string(action.action.value)] = record_data!(d, M.𝒮.F.D["node"][string(action.action.value)])
                        M.𝒮.F.D_full["node"][string(action.action.value)] = record_data!(d_full, M.𝒮.F.D_full["node"][string(action.action.value)])
                    else
                        # record_data(d,joinpath(abspath(@__DIR__), "data", "edge_$(action.action.value).csv"))
                        M.𝒮.F.D["edge"][string(action.action.value)] = record_data!(d, M.𝒮.F.D["edge"][string(action.action.value)])
                        M.𝒮.F.D_full["edge"][string(action.action.value)] = record_data!(d_full, M.𝒮.F.D_full["edge"][string(action.action.value)])
                    end
                end
            elseif action.l == 2 #|| (action.l == 1 && !M.flags[M.𝒮.D.SIndex[state.state]][M.𝒮.D.AIndex[action.action]])
                σ = generate_feedback(state, M.𝒮.W, action)
                if i == m
                    y = (σ == '∅') ? 1 : 0
                    d = hcat(get_state_features(M, state.state), 2, y)
                    d_full = hcat(get_full_state_features(M, state.state), 2, y)
                    if typeof(state.state) == NodeState
                        # record_data(d,joinpath(abspath(@__DIR__), "data", "node_$(action.action.value).csv"))
                        M.𝒮.F.D["node"][string(action.action.value)] = record_data!(d, M.𝒮.F.D["node"][string(action.action.value)])
                        M.𝒮.F.D_full["node"][string(action.action.value)] = record_data!(d_full, M.𝒮.F.D_full["node"][string(action.action.value)])
                    else
                        # record_data(d,joinpath(abspath(@__DIR__), "data", "edge_$(action.action.value).csv"))
                        M.𝒮.F.D["edge"][string(action.action.value)] = record_data!(d, M.𝒮.F.D["edge"][string(action.action.value)])
                        M.𝒮.F.D_full["edge"][string(action.action.value)] = record_data!(d_full, M.𝒮.F.D_full["edge"][string(action.action.value)])
                    end
                end
            end
            # println("received feedback: $σ")
            if σ != '∅'
                override_rate_records_per_ep[state.state][2] += 1
                if i == m
                    signal_count += 1
                end
            end
            episode_cost += C(M, s, a)
            if σ == '⊖'
                block_transition!(M, state, action)
                state′ = CASstate(state.state, '⊖')
                # M.s₀ = state
                # L = solve_model(M)
                delete!(L.solved, s)
                update_world_state!(M.𝒮.W, state.state, action.action, state′.state)
                state = state′
                if typeof(state.state) == NodeState
                    ISR = Tuple([getproperty(M.𝒮.W, f) for f in M.𝒮.D.F_active["node"]
                                   if hasproperty(M.𝒮.W, f)])
                    state = CASstate(NodeState(state.state.id, state.state.p, state.state.o, state.state.v, state.state.θ, ISR), state.σ)
                else
                    ISR = Tuple([getproperty(M.𝒮.W, f) for f in M.𝒮.D.F_active["edge"]
                                   if hasproperty(M.𝒮.W, f)])
                    state = CASstate(EdgeState(state.state.u, state.state.v, state.state.θ, state.state.o, state.state.l, ISR), state.σ)
                end
                continue
            end
            if action.l == 0 || σ == '⊘'
                state′ = M.S[M.T[s][a][1][1]]
            else
                state′ = generate_successor(M.𝒮.D, state, action, σ)
            end
            update_world_state!(M.𝒮.W, state.state, action.action, state′.state)
            state = state′
            if typeof(state.state) == NodeState
                ISR = Tuple([getproperty(M.𝒮.W, f) for f in M.𝒮.D.F_active["node"]
                               if hasproperty(M.𝒮.W, f)])
                state = CASstate(NodeState(state.state.id, state.state.p, state.state.o, state.state.v, state.state.θ, ISR), state.σ)
            else
                ISR = Tuple([getproperty(M.𝒮.W, f) for f in M.𝒮.D.F_active["edge"]
                               if hasproperty(M.𝒮.W, f)])
                state = CASstate(EdgeState(state.state.u, state.state.v, state.state.θ, state.state.o, state.state.l, ISR), state.σ)
            end
            # println(σ, "     | succ state |      ", state)
            if terminal(M, state) || episode_cost > 500.0
                if episode_cost > 500.0
                    @infiltrate
                end
                break
            end
        end

        push!(c, episode_cost)
        M.T = T_base
    end
    push!(override_rate_records, override_rate_records_per_ep)
    println("Total cumulative reward: $(round(mean(c);digits=4)) ⨦ $(std(c))")
    return mean(c), std(c), signal_count, (actions_at_competence / actions_taken), (abs(mean(c) - expected_cost)/expected_cost)
end

function run_episodes(M, C)
    println("Starting")
    # Tracking information
    los, los_r = Vector{Float64}(), Vector{Float64}()
    costs, costs2 = Vector{Float64}(), Vector{Float64}()
    stds, stds2 = Vector{Float64}(), Vector{Float64}()
    cost_errors, cost_errors2 = Vector{Float64}(), Vector{Float64}()
    fixed_task_costs, fixed_task_costs2 = Vector{Float64}(), Vector{Float64}()
    signal_counts, signal_counts2 = Vector{Int}(), Vector{Int}()
    signal_counts_per_10, signal_counts_per_10_2 = Vector{Int}(), Vector{Int}()
    lo_function_of_signal_count = Vector{Tuple{Int, Float64}}()
    lo_function_of_signal_count2 = Vector{Tuple{Int, Float64}}()
    route_records = Dict{Int, Dict{String, Any}}()
    total_signals_received, total_signals_received2 = 0, 0
    expected_task_costs = Vector{Float64}()
    discrims = []
    results = []
    for i=1:200
        println(i)
        initialize_random_start!(C.𝒮.W)

        # Set a random route.
        route, (init, goal) = rand(fixed_routes)
        set_route(M, C, init, goal)
        generate_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)

        println(i, "  |  Task: ", route)
        ℒ = solve_model(C)
        push!(expected_task_costs, ℒ.V[C.SIndex[C.s₀]])
        c, standard_dev, signal_count, percent_lo, error = simulate(C, ℒ, 10)
        total_signals_received += signal_count

        # ℒ2 = solve_model(M)
        # c2, std2, signal_count2, percent_lo2, error2 = simulate(M, C, ℒ2, 10)
        # total_signals_received2 += signal_count2

        # Per episode record keeping.
        push!(costs, c)
        push!(stds, standard_dev)
        push!(cost_errors, error)
        push!(signal_counts, total_signals_received)
        push!(lo_function_of_signal_count, (total_signals_received, percent_lo))

        # push!(costs2, c2)
        # push!(stds2, std2)
        # push!(cost_errors2, error2)
        # push!(signal_counts2, total_signals_received2)
        # push!(lo_function_of_signal_count2, (total_signals_received2, percent_lo2))

        # Per 10 episode record keeping (high compute).
        if i == 1 || i%10 == 0
            lo, lo_r = compute_level_optimality(C, ℒ)
            push!(los, lo)
            push!(los_r, lo_r)
            push!(signal_counts_per_10, total_signals_received)
            push!(signal_counts_per_10_2, total_signals_received2)

            # Fixed route results
            # set_route(M, C, 12, 7)
            # generate_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)
            # L = solve_model(C)
            # push!(expected_task_costs, L.V[C.SIndex[C.s₀]])
            # L2 = solve_model(M)
            # c, std, signal_count, percent_lo, error = simulate(C, L, 1000)
            # c2, std2, signal_count2, percent_lo2, error2 = simulate(M, C, L2, 1000)
            # push!(fixed_task_costs, c)
            # push!(fixed_task_costs2, c2)


            # if i != 1
            #     if !isempty(M.F_inactive)
            #         candidates = find_candidates(C)
            #         if !isempty(candidates)
            #             candidate = sample(candidates)
            #             discriminator = get_discriminator(C, candidate, 3)
            #             if discriminator != -1
            #                 @infiltrate
            #                 println("Added discriminator $discriminator.")
            #                 push!(discrims, (i, discriminator))
            #                 update_features!(M, discriminator)
            #                 for action in M.A
            #                     update_data!(C, action, "node")
            #                     if action.value in ['↑', '⤉']
            #                         update_data!(C, action, "edge")
            #                     end
            #                 end
            #                 save_data(C.𝒮.F.D)
            #                 save_full_data(C.𝒮.F.D_full)
            #
            #                 # M = build_model(C.𝒮.W, M.F_active, M.F_inactive)
            #                 # C = build_cas(M, C.𝒮.W, [0,1,2,3], ['⊕', '⊖', '⊘', '∅'])
            #                 println("Rebuilding models...")
            #                 build_model!(M, C.𝒮.W)
            #                 build_cas!(C)
            #             end
            #         end
            #     end
            # end
        end

        # if i ==1 || i%100 == 0
        #     route_records[i] = Dict{String, Any}()
        #     for k in keys(fixed_routes)
        #         if !haskey(route_records[i], k)
        #             route_records[i][k] = Dict()
        #         end
        #         init, goal = fixed_routes[k]
        #         println("Getting route: $init --> $goal")
        #         set_route(M, C, init, goal)
        #         generate_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)
        #         L = solve_model(C)
        #         route = get_route(M, C, L)
        #         route_records[i][k]["route"] = route
        #         route_records[i][k]["expected cost"] = L.V[C.SIndex[C.s₀]]
        #     end
        # end

        # Update model
        # update_feedback_profile!(C)
        # update_autonomy_profile!(C, ℒ)
        save_data(C.𝒮.F.D)
        generate_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)

        # results = [costs, costs2, stds, stds2, expected_task_costs, cost_errors, cost_errors2, los, los_r, lo_function_of_signal_count, lo_function_of_signal_count2, signal_counts, signal_counts2, fixed_task_costs, fixed_task_costs2, route_records]
        results = [costs, stds, expected_task_costs, cost_errors, los, los_r, lo_function_of_signal_count, signal_counts, fixed_task_costs, route_records, discrims]
        save(joinpath(abspath(@__DIR__), "results.jld"), "results", results)

        x = signal_counts
        los_a = [v[2] for v in lo_function_of_signal_count]
        los_a = append!([los_a[1]], los_a[10:10:end])

        g = scatter(signal_counts_per_10, [los los_r los_a], legend=:topleft, ylims=(0.,1.), xlabel="Signals Received", ylabel="Level Optimality", label = ["All States" "Reachable" "Visited"])
        savefig(g, joinpath(abspath(@__DIR__), "plots", "level_optimality_by_signal_count.png"))

        g2 = scatter(x, cost_errors, xlabel="Signals Received", legend=:topleft, ylabel="%Error")
        savefig(g2, joinpath(abspath(@__DIR__), "plots", "percent_error.png"))

        g3 = scatter(x, stds, xlabel="Signals Received", legend=:topleft, ylabel="Reliability")
        savefig(g3, joinpath(abspath(@__DIR__), "plots", "standard_devs.png"))

        # g4 = scatter(signal_counts_per_10, fixed_task_costs, xlabel="Episode", ylabel="Average Cost to Goal")
        # savefig(g4, joinpath(abspath(@__DIR__), "plots", "fixed_task_costs.png"))
    end
    save_autonomy_profile(C.𝒮.A.κ)
    save(joinpath(abspath(@__DIR__), "override_records.jld"), "override_records", override_rate_records)

    println(costs)
    println(stds)
    println(cost_errors)
    println(los)
    println(los_r)
    println(lo_function_of_signal_count)
    println(signal_counts)
    println(fixed_task_costs)

    return results
end

W = get_random_world_state()
M = build_model(W)
init_data(M)
C = build_cas(M, W, [0,1,2,3], ['⊕', '⊖', '⊘', '∅'])
override_rate_records = Vector{Dict{DomainState, Array{Int}}}()
results = run_episodes(M, C)
simulate(C, L, 1)
# results2 = run_episodes(M, C)
# simulate(M, L2, 10)
# L = solve_model(C)
# simulate(C, L, 1)
# save(joinpath(abspath(@__DIR__), "override_records.jl"), "override_records", override_rate_records)
# D = Dict{String, Dict{String, DataFrame}}()
#
# D["node"] = Dict{String, DataFrame}()
# for a in ["↑", "→", "↓", "←", "⤉"]
#     D["node"][a] = DataFrame(CSV.File(joinpath(abspath(@__DIR__), "data", "node_$a.csv")))
# end
#
# results = load(joinpath(abspath(@__DIR__), "data_eps+", "results.jld"), "results")
# signal_counts_per_10 = cat(results[12][1], results[12][10:10:end], dims=1)
# savefig(scatter(signal_counts_per_10, [results[8] results[9]],
#         xlims=[0, 250], ylims=[0, 1.0], legend=:topleft,
#         xlabel="Signal Count", ylabel="Level-Optimality", label=["All States" "Reachable"]),
#         joinpath(abspath(@__DIR__), "data_eps+", "plots", "level_optimality_by_signal_count_eps+.png"))
#

results1 = load(joinpath(abspath(@__DIR__), "Experiment 1", "new_person_untrusting", "isr", "results.jld"), "results")
results2 = load(joinpath(abspath(@__DIR__), "Experiment 1", "new_person_untrusting", "normal", "results.jld"), "results")

costs1 = results1[1]
costs2 = results2[1]

exp_costs1 = results1[3]
exp_costs2 = results2[3]

errors1 = (costs1 .- exp_costs1) ./ costs1
