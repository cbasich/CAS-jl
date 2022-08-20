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
            actions_at_competence += (1 == competence(state, action))
            # println("$i   |   Taking action $action in state $state.")
            σ = generate_feedback(CASstate(state, '∅'), CASaction(action, 2))
            episode_cost += C.C(C, C.SIndex[CASstate(state, σ)], C.AIndex[CASaction(action, 2)])

            if σ == '⊘'
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
            actions_at_competence += (action.l == competence(state.state, M.𝒮.W, action.action))
            # println("$i   |   Taking action $action in state $state.")
            # println((action.l == competence(state.state, M.𝒮.W, action.action)))
            if action.l == 0 || action.l == 2
                σ = '∅'
            else
                σ = generate_feedback(state.state, M.𝒮.W, action.action)
                if i == m
                    y = (σ == '∅')
                    d = hcat(get_state_features(M, state.state), y)
                    M.𝒮.F.D[string(action.action.value)] = record_data!(d, M.𝒮.F.D[string(action.action.value)])

                    d_full = hcat(get_full_state_features(M, state.state), y)
                    M.𝒮.F.D_full[string(action.action.value)] = record_data!(d_full, M.𝒮.F.D_full[string(action.action.value)])
                end
            end
            # println("received feedback: $σ")
            if σ == '⊘'
                override_rate_records_per_ep[state.state][2] += 1
                if i == m
                    # println("override detected at state:     $state      |     action: $action")
                    signal_count += 1
                end
            end
            episode_cost += C(M, s, a)
            if action.l == 0
                state = CASstate(DomainState(4, 0, 0, state.state.ISR), '∅')
            elseif σ == '⊘'
                state = CASstate(DomainState(4, 0, 0, state.state.ISR), '⊘')
            else
                state = generate_successor(M.𝒮.D, state, action, σ)
            end
            # println(σ, "     | succ state |      ", state)
            if terminal(M, state)
                break
            end
        end

        push!(c, episode_cost)
    end
    # println("Total cumulative reward: $(round(mean(c);digits=4)) ⨦ $(std(c))")
    push!(override_rate_records, override_rate_records_per_ep)
    return mean(c), std(c), signal_count, (actions_at_competence / actions_taken), (abs(mean(c) - expected_cost)/expected_cost)
end

function run_episodes(M, C, L)
    println("Starting")

    # Tracking information
    los = Vector{Float64}()
    los_r = Vector{Float64}()
    costs = Vector{Float64}()
    stds = Vector{Float64}()
    cost_errors = Vector{Float64}()
    expected_task_costs = Vector{Float64}()
    signal_counts = Vector{Int}()
    signal_counts_per_10 = Vector{Int}()
    lo_function_of_signal_count = Vector{Tuple{Int, Float64}}()
    discrims = Vector{Any}()
    # override_rate_records_by_ep = Vector{Dict{DomainState, Array{Int}}}()
    total_signals_received = 0
    results = []
    for i=1:1000
        println(i)
        set_random_world_state!(C.𝒮.W)
        M.s₀ = DomainState(0, -1, false, Tuple([getproperty(C.𝒮.W, f)
                           for f in M.F_active if hasproperty(C.𝒮.W, f)]))
        C.s₀ = CASstate(M.s₀, '∅')
        generate_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)
        ℒ = solve_model(C)
        c, std, signal_count, percent_lo, error = simulate(C, ℒ, 100)
        total_signals_received += signal_count

        # Per episode record keeping.
        push!(expected_task_costs, ℒ.V[C.SIndex[C.s₀]])
        push!(costs, c)
        push!(stds, std)
        push!(cost_errors, error)
        push!(signal_counts, total_signals_received)
        push!(lo_function_of_signal_count, (total_signals_received, percent_lo))

        # Per 10 episode record keeping (high compute).
        if i == 1 || i%5 == 0
            lo, lo_r = compute_level_optimality(C, ℒ)
            push!(los, lo)
            push!(los_r, lo_r)
            # println(last(los))
            push!(signal_counts_per_10, total_signals_received)
            # push!(override_rate_records_by_ep, deepcopy(override_rate_records))

            # for (a, data) in C.𝒮.F.D
            #     record_data(data,joinpath(abspath(@__DIR__), "data", "$a.csv"), false)
            # end
            if i%10 == 0
                if !isempty(M.F_inactive)
                    candidates = find_candidates(C)
                    if !isempty(candidates)
                        candidate = sample(candidates)
                        discriminator = get_discriminator(C, candidate, 3)
                        if discriminator != -1
                            push!(discrims, i)
                            println("Adding discriminator $discriminator")
                            update_features!(M, discriminator)
                            for action in M.A
                                update_data!(C, action)
                            end
                            save_data(C.𝒮.F.D)
                            save_full_data(C.𝒮.F.D_full)

                            # M = build_model(C.𝒮.W, M.F_active, M.F_inactive)
                            # C = build_cas(M, C.𝒮.W, [0,1,2], ['∅', '⊘'])
                            build_model!(M, C.𝒮.W)
                            build_cas!(C)
                        end
                    end
                end
            end
        end

        # Update model
        # println("Updating Model.")
        update_feedback_profile!(C)
        update_autonomy_profile!(C, ℒ)


        results = [costs, stds, cost_errors, los, los_r, lo_function_of_signal_count, signal_counts, expected_task_costs, discrims]
        save_object(joinpath(abspath(@__DIR__), "results.jld"), results)

        x = signal_counts

        los_a = [x[2] for x in lo_function_of_signal_count]
        los_a = append!([los_a[1]], los_a[5:5:end])

        g = scatter(signal_counts_per_10, [los los_r los_a], alpha=0.6, legend=:topleft, ylims=(0.,1.), xlabel="Signals Received", ylabel="Level Optimality", label = ["All States" "Reachable" "Visited" ])
        savefig(g, joinpath(abspath(@__DIR__), "plots", "level_optimality_by_signal_count.png"))
        #
        g2 = scatter(x, cost_errors, legend=:topleft, xlabel="Signals Received", ylabel="%Error")
        savefig(g2, joinpath(abspath(@__DIR__), "plots", "percent_error.png"))

        g3 = scatter(x, stds, legend=:topleft, xlabel="Signals Received", ylabel="Reliability")
        savefig(g3, joinpath(abspath(@__DIR__), "plots", "reliability.png"))

        g4 = scatter(x, costs, legend=:topleft, xlabel="Signals Received", ylabel="Cost to Goal")
        savefig(g4, joinpath(abspath(@__DIR__), "plots", "task_cost.png"))

        g5 = scatter(x, expected_task_costs, legend=:topleft, xlabel="Signals Received", ylabel="Expected Cost to Goal")
        savefig(g5, joinpath(abspath(@__DIR__), "plots", "expected_task_cost.png"))

        # g6 = scatter(x, smooth_data(expected_task_costs .- costs,5), legend=:topleft, xlabel="Signals Received", ylabel="Expected Cost to Goal")
        # savefig(g6, joinpath(abspath(@__DIR__), "plots", "cost_error.png"))
    end
    save_data(C.𝒮.F.D)
    save_full_data(C.𝒮.F.D_full)
    save_autonomy_profile(C.𝒮.A.κ)
    save(joinpath(abspath(@__DIR__), "override_records.jld"), "override_records", override_rate_records)

    println(costs)
    println(stds)
    println(cost_errors)
    println(los)
    println(lo_function_of_signal_count)
    println(signal_counts)
    println(expected_task_costs)

    return results
end

W = get_random_world_state()
M = build_model(W)
init_data(M)
C = build_cas(M, W, [0,1,2], ['∅', '⊘'])
L = solve_model(C)
override_rate_records = Vector{Dict{DomainState, Array{Int}}}()
results = run_episodes(M, C, L)


simulate(C, L, 1)

results = load(joinpath(abspath(@__DIR__), "person_rush", "ISR", "results.jld"), "results")
results2 = load(joinpath(abspath(@__DIR__), "person_rush", "normal", "results.jld"), "results")

ISR_los_active = [x[2] for x in results[6][1:10:100]]
norm_los_active = [x[2] for x in results2[6][1:10:500]]

scatter([ISR_los_active norm_los_active results[4][1:50] results2[4][1:50]])
scatter([results[4] results2[4]])

results = load(joinpath(abspath(@__DIR__), "person_rush", "ISR", "results.jld"), "results")
signal_counts_per_10 = cat(results[6][1], results[6][10:10:end], dims=1)
savefig(scatter(signal_counts_per_10, [results[4] results[5] results[6]],
        xlims=[0, 50], ylims=[0, 1.0], legend=:topleft,
        xlabel="Signal Count", ylabel="Level-Optimality", label=["All States" "Reachable"]),
        joinpath(abspath(@__DIR__),  "person_rush", "ISR", "plots", "level_optimality_by_signal_count.png"))

x = results[7][6:end]
exp_cost = smooth_data(results[8],5)
savefig(scatter(exp_cost, xlabel="Signal Count", ylabel="Expected Cost", label=""),
        joinpath(abspath(@__DIR__), "plots", "exp_task_cost.png"))

task_cost = smooth_data(results[1],2)
savefig(scatter(task_cost, xlabel="Signal Count", ylabel="Expected Cost", label=""),
        joinpath(abspath(@__DIR__), "plots", "task_cost_per_episode.png"))
