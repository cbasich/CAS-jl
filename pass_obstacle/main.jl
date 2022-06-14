using Profile,ProfileView
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
            actions_at_competence += (action.l == competence(state.state, action.action))
            # println("$i   |   Taking action $action in state $state.")
            if action.l == 0 || action.l == 2
                σ = '∅'
            else
                σ = generate_feedback(state.state, action.action)
                if i == m
                    y = (σ == '∅')
                    d = hcat(get_state_features(state.state), y)
                    # record_data(d,joinpath(abspath(@__DIR__), "data", "$(action.action.value).csv"))
                    record_data!(d, M.𝒮.F.D[string(action.action.value)])
                end
            end
            # println("received feedback: $σ")
            if σ == '⊘'
                override_rate_records_per_ep[state.state][2] += 1
                if i == m
                    println("override detected at state:     $state      |     action: $action")
                    signal_count += 1
                end
            end
            episode_cost += C(M, s, a)
            if action.l == 0 || σ == '⊘'
                state = M.S[M.T[s][a][1][1]]
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
    costs = Vector{Float64}()
    stds = Vector{Float64}()
    cost_errors = Vector{Float64}()
    expected_task_costs = Vector{Float64}()
    signal_counts = Vector{Int}()
    signal_counts_per_10 = Vector{Int}()
    lo_function_of_signal_count = Vector{Tuple{Int, Float64}}()
    # override_rate_records_by_ep = Vector{Dict{DomainState, Array{Int}}}()
    total_signals_received = 0
    results = []
    for i=1:1000
        println(i)
        ℒ = solve_model(C)
        c, std, signal_count, percent_lo, error = simulate(C, ℒ, 100)
        total_signals_received += signal_count

        # Per episode record keeping.
        push!(costs, c)
        push!(stds, std)
        push!(cost_errors, error)
        push!(signal_counts, total_signals_received)
        push!(lo_function_of_signal_count, (total_signals_received, percent_lo))

        # Per 10 episode record keeping (high compute).
        if i == 1 || i%10 == 0
            push!(los, compute_level_optimality(C, ℒ))
            println(last(los))
            push!(signal_counts_per_10, total_signals_received)
            # push!(override_rate_records_by_ep, deepcopy(override_rate_records))

            # for (a, data) in C.𝒮.F.D
            #     record_data(data,joinpath(abspath(@__DIR__), "data", "$a.csv"), false)
            # end
        end

        # Update model
        # println("Updating Model.")
        update_feedback_profile!(C)
        update_autonomy_profile!(C, ℒ)
        generate_transitions!(C.𝒮.D, C.𝒮.A, C.𝒮.F, C, C.S, C.A, C.G)

        results = [costs, stds, cost_errors, los, lo_function_of_signal_count, signal_counts, expected_task_costs]
        save(joinpath(abspath(@__DIR__), "results.jld"), "results2", results)

        x = signal_counts

        g = scatter(signal_counts_per_10, los, xlabel="Signals Received", ylabel="Level Optimality")
        savefig(g, joinpath(abspath(@__DIR__), "plots", "level_optimality_by_signal_count.png"))
        #
        g2 = scatter(x, cost_errors, xlabel="Signals Received", ylabel="%Error")
        savefig(g2, joinpath(abspath(@__DIR__), "plots", "percent_error.png"))

        g3 = scatter(x, stds, xlabel="Signals Received", ylabel="Reliability")
        savefig(g3, joinpath(abspath(@__DIR__), "plots", "percent_error.png"))

        g4 = scatter(x, costs, xlabel="Episode", ylabel="Cost to Goal")
        savefig(g4, joinpath(abspath(@__DIR__), "plots", "task_cost.png"))
    end
    save_data(C.𝒮.F.D)
    save_autonomy_profile(C.𝒮.A.κ)

    println(costs)
    println(stds)
    println(cost_errors)
    println(los)
    println(lo_function_of_signal_count)
    println(signal_counts)
    println(expected_task_costs)

    return results
end

init_data()
M = build_model()
C = build_cas(M, [0,1,2], ['∅', '⊘'])
L = solve_model(C)
override_rate_records = Vector{Dict{DomainState, Array{Int}}}()
results = run_episodes(M, C, L)
