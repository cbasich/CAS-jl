using Profile,ProfileView,JLD2
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
            σ = generate_feedback(CASstate(state, '∅'), CASaction(action, 2), C.𝒮.F.ϵ)
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
            actions_taken += 1
            actions_at_competence += (action.l == competence(state.state, action.action))
            println("$i   |   Taking action $action in state $state.")
            if action.l == 0 || action.l == 3
                σ = '∅'
            elseif action.l == 1
                σ = generate_feedback(state, action, M.𝒮.F.ϵ)
                if i == m
                    y = (σ == '⊕') ? 1 : 0
                    d = hcat(get_state_features(state.state), 1, y)
                    if typeof(state.state) == NodeState
                        # record_data(d,joinpath(abspath(@__DIR__), "data", "node_$(action.action.value).csv"))
                        M.𝒮.F.D["node"][string(action.action.value)] = record_data!(d, M.𝒮.F.D["node"][string(action.action.value)])
                    else
                        # record_data(d,joinpath(abspath(@__DIR__), "data", "edge_$(action.action.value).csv"))
                        M.𝒮.F.D["edge"][string(action.action.value)] = record_data!(d, M.𝒮.F.D["edge"][string(action.action.value)])
                    end
                end
            elseif action.l == 2 #|| (action.l == 1 && !M.flags[M.𝒮.D.SIndex[state.state]][M.𝒮.D.AIndex[action.action]])
                σ = generate_feedback(state, action, M.𝒮.F.ϵ)
                if i == m
                    y = (σ == '∅') ? 1 : 0
                    d = hcat(get_state_features(state.state), 2, y)
                    if typeof(state.state) == NodeState
                        # record_data(d,joinpath(abspath(@__DIR__), "data", "node_$(action.action.value).csv"))
                        M.𝒮.F.D["node"][string(action.action.value)] = record_data!(d, M.𝒮.F.D["node"][string(action.action.value)])
                    else
                        # record_data(d,joinpath(abspath(@__DIR__), "data", "edge_$(action.action.value).csv"))
                        M.𝒮.F.D["edge"][string(action.action.value)] = record_data!(d, M.𝒮.F.D["edge"][string(action.action.value)])
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
                state = CASstate(state.sh, state.state, '⊖')
                # M.s₀ = state
                # L = solve_model(M)
                delete!(L.solved, s)
                continue
            end
            if action.l == 0 || σ == '⊘'
                # state = M.S[M.T[s][a][1][1]]
                temp = M.T[s][a]
                state = M.S[sample(first.(temp), aweights(last.(temp)))]
            else
                state = generate_successor(M.𝒮.D, state, action, σ)
            end
            # println(σ, "     | succ state |      ", state)
            if terminal(M, state) #|| episode_cost > 100.0
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
    mean_costs, mean_costs2 = Vector{Float64}(), Vector{Float64}()
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
    results = []
    for i=1:1000
        # Set a random route.
        route, (init, goal) = rand(fixed_routes)
        w = generate_random_world_state()
        set_route(M, C, init, goal, w)
        generate_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)

        println(i, "  |  Task: ", route)
        ℒ = solve_model(C)
        push!(mean_costs, ℒ.V[C.SIndex[C.s₀]])
        c, std, signal_count, percent_lo, error = simulate(C, ℒ, 10)
        total_signals_received += signal_count

        # ℒ2 = solve_model(M)
        # push!(mean_costs2, ℒ2.V[M.SIndex[M.s₀]])
        # c2, std2, signal_count2, percent_lo2, error2 = simulate(M, C, ℒ2, 10)
        # total_signals_received2 += signal_count2

        # Per episode record keeping.
        push!(costs, c)
        push!(stds, std)
        push!(cost_errors, error)
        push!(signal_counts, total_signals_received)
        push!(lo_function_of_signal_count, (total_signals_received, percent_lo))

        # push!(costs2, c2)
        # push!(stds2, std2)
        # push!(cost_errors2, error2)
        # push!(signal_counts2, total_signals_received2)
        # push!(lo_function_of_signal_count2, (total_signals_received2, percent_lo2))

        # Per 10 episode record keeping (high compute).
        # if i == 1 || i%10 == 0
        #     lo, lo_r = compute_level_optimality(C, ℒ)
        #     push!(los, lo)
        #     push!(los_r, lo_r)
        #     push!(signal_counts_per_10, total_signals_received)
        #     push!(signal_counts_per_10_2, total_signals_received2)
        #
        #     # Fixed route results
        #     set_route(M, C, 12, 7)
        #     generate_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)
        #     L = solve_model(C)
        #     push!(expected_task_costs, L.V[C.SIndex[C.s₀]])
        #     L2 = solve_model(M)
        #     c, std, signal_count, percent_lo, error = simulate(C, L, 100)
        #     c2, std2, signal_count2, percent_lo2, error2 = simulate(M, C, L2, 100)
        #     push!(fixed_task_costs, c)
        #     push!(fixed_task_costs2, c2)
        # end

        # if i == 1 || i%100 == 0
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
        update_feedback_profile!(C)
        update_autonomy_profile!(C, ℒ)
        save_data(C.𝒮.F.D)
        # generate_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)
        # set_consistency(C.𝒮.F, min(1.0, C.𝒮.F.ϵ+0.01))

        # results = [costs, costs2, stds, stds2, expected_task_costs, cost_errors, cost_errors2, los, los_r, lo_function_of_signal_count, lo_function_of_signal_count2, signal_counts, signal_counts2, route_records]

        results = [costs, costs2, stds, expected_task_costs, cost_errors, los, los_r, lo_function_of_signal_count, signal_counts, signal_counts2, fixed_task_costs, fixed_task_costs2]

        # jldsave(joinpath(abspath(@__DIR__), "results.jld2"); results)
        save_object(joinpath(abspath(@__DIR__), "results.jld2"), results)


        x = signal_counts
        los_a = [v[2] for v in lo_function_of_signal_count]
        los_a = append!([los_a[1]], los_a[10:10:end])

        g = scatter(signal_counts_per_10, [los los_r], legend=:topleft, ylims=(0.,1.), xlabel="Signals Received", ylabel="Level Optimality", label = ["All States" "Reachable"])
        savefig(g, joinpath(abspath(@__DIR__), "plots", "level_optimality_by_signal_count.png"))

        g1 = plot([los los_r], legend=:topleft, ylims=(0.,1.), xlabel="Episode", ylabel="Level Optimality", label = ["All States" "Reachable"])
        savefig(g1, joinpath(abspath(@__DIR__), "plots", "level_optimality_by_episode.png"))

        # g2 = scatter(x, [cost_errors cost_errors2], xlabel="Signals Received", ylabel="%Error", label = ["CAS" "No CAS"])
        # savefig(g2, joinpath(abspath(@__DIR__), "plots", "percent_error.png"))

        # g3 = scatter(x, [stds stds2], xlabel="Signals Received", ylabel="Reliability", label = ["CAS" "No CAS"])
        # savefig(g3, joinpath(abspath(@__DIR__), "plots", "standard_devs.png"))

        g4 = scatter(signal_counts_per_10, [fixed_task_costs fixed_task_costs2], xlabel="Signal Count", ylabel="Average Cost to Goal", label = ["CAS" "No CAS"])
        savefig(g4, joinpath(abspath(@__DIR__), "plots", "fixed_task_costs.png"))

        # task_errors = 100.0 .* ((costs .- expected_task_costs)./costs)
        # g5 = plot(task_errors, xlabel="Episode", ylabel="% Error")
        # savefig(g5, joinpath(abspath(@__DIR__), "plots", "cost_errors.png"))
    end
    save_autonomy_profile(C.𝒮.A.κ)
    # JLD2.save(joinpath(abspath(@__DIR__), "override_records.jld2"), "override_records", override_rate_records)
    save_object(joinpath(abspath(@__DIR__), "override_records.jld2"), override_rate_records)

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

init_data()
# CAS_vec = Vector{CASSP}()
M = build_model()
C = build_cas(M, [0,1,2,3], ['⊕', '⊖', '⊘', '∅'])
# set_route(M, C, 12, 10)
# generate_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)
# L = LRTDPsolver(C, 10000., 100, .001, Dict{Int, Int}(),
                 # false, Set{Int}(), zeros(length(C.S)), zeros(length(C.A)))
# L2 = LRTDPsolver(M, 10000., 100, .001, Dict{Int, Int}(),
                 # false, Set{Int}(), zeros(length(M.S)), zeros(length(M.A)))
override_rate_records = Vector{Dict{DomainState, Array{Int}}}()
results = run_episodes(M, C)
results = load_object(joinpath(abspath(@__DIR__), "experiment_7_21_22", "eps100", "results.jld2"))
los_a = results[5]
los_r = results[6]
# los_v = [x[2] for x in results[7]]
# los_v = append!([los_v[1]], los_v[10:10:end])
signal_count = results[8]
signal_count = append!([signal_count[1]], signal_count[10:10:end])


savefig(scatter(signal_count, [los_a los_r],
             xlims=[0, 80], ylims=[0, 1.0], legend=:bottomright,
             xlabel="Signals Received", ylabel="Level Optimality",
             label=["All States" "Reachable" "Visited"]),
             joinpath(abspath(@__DIR__), "plots", "level_optimality_by_signal_count_eps100.png"))

savefig(plot([results[8][1:100] results[9][1:100]],
             xlims=[0, 100], ylims=[0, 1000], legend=:topleft,
             xlabel="Episode", ylabel="Signals Received", label=["CAS" "No CAS"]),
             joinpath(abspath(@__DIR__), "plots", "signal_count_comparison_eps100.png"))

savefig(plot([results[10][1:100] .- results[11][1:100]],
            legend=:topleft,
            xlabel="Episode", ylabel="Average Cost", label=["CAS" "No CAS"]),
            joinpath(abspath(@__DIR__), "plots", "avg_cost_comp_eps100.png"))

exp_cost = results[3][12:11:end]
exp_cost = append!([results[3][2]], exp_cost)
savefig(scatter(signal_count, results[4], xlabel="Signals Received", ylabel="Expected Cost", label="", size=(600,400)),
            joinpath(abspath(@__DIR__), "plots", "expected_cost_eps90.png"))

savefig(scatter(signal_count, [results[11] results[12]],
                xlabel="Signals Received", ylabel="Avergage Cost",
                labels=["CAS" "No CAS"], size=(600,400)),
                joinpath(abspath(@__DIR__), "plots", "cost_comparison90.png"))

results2 = run_episodes(M, C)
simulate(M, L2, 10)
L = solve_model(C)
simulate(C, L, 1)
save(joinpath(abspath(@__DIR__), "override_records.jl"), "override_records", override_rate_records)
D = Dict{String, Dict{String, DataFrame}}()

D["node"] = Dict{String, DataFrame}()
for a in ["↑", "→", "↓", "←", "⤉"]
    D["node"][a] = DataFrame(CSV.File(joinpath(abspath(@__DIR__), "data", "node_$a.csv")))
end

results2 = @load joinpath(abspath(@__DIR__), "results.jld2") results
# results2 = load(joinpath(abspath(@__DIR__), "results.jld"), "results")
signal_counts_per_10 = cat(results[7][1], results[7][10:10:end], dims=1)
savefig(scatter(signal_counts_per_10, [results[5] results[6]],
        xlims=[0, 250], ylims=[0, 1.0], legend=:topleft,
        xlabel="Signal Count", ylabel="Level-Optimality", label=["All States" "Reachable"]),
        joinpath(abspath(@__DIR__), "eps100_newestcost2", "plots", "level_optimality_by_signal_count_eps+.png"))


routes = last(results)
starting_routes = routes[1]
ending_routes = routes[400]

init_route = last(results)[1]
end_route = last(results)[400]

for k in keys(init_route)
    if init_route[k]["route"] != end_route[k]["route"]
    end
        println(k, ": ", init_route[k], "  ⟶  ", end_route[k])
end

total_overrides = override_rate_records[1]
for i=2:length(override_rate_records)
    for k in keys(override_rate_records[i])
        if haskey(total_overrides, k)
            total_overrides[k][1] += override_rate_records[i][k][1]
            total_overrides[k][2] += override_rate_records[i][k][2]
        else
            total_overrides[k] = override_rate_records[i][k]
        end
    end
end
override_rates = Dict(k => (v[2]/v[1]) for (k,v) in total_overrides if (v[1] > 50 && v[2] > 0))

results = load(joinpath(abspath(@__DIR__), "results.jld"), "results90")
results2 = load(joinpath(abspath(@__DIR__), "results3.jld"), "results3")
los = cat(results[7], results2[7], dims=1)[1:130]
los_r = cat(results[8], results2[8], dims=1)[1:130]
signal_counts_per_10 = cat(results[11], (363 .+ results2[11]), dims=1)#[1:10:end]

exp_cost1 = cat(results[13], results2[13], dims=1)
exp_cost2 = cat(results[14], results2[14], dims=1)
g4 = scatter(signal_counts_per_10, [exp_cost1 exp_cost2], xlabel="Signals", ylabel="Expected Cost to Goal")
