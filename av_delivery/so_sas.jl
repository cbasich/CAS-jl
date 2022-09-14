struct SOSASAction
    operator::Int
end

struct SOSAS
    AV
    H
    S
    A
    T
    C
    s₀
    G
    SIndex
    AIndex
end
function SOSAS(AV, H, S, A, T, C, s₀, G)
    SIndex, AIndex = generate_index_dicts(S, A)
    return SOSAS(AV, H, S, A, T, C, s₀, G, SIndex, AIndex)
end

function generate_index_dicts(S::Vector{CASstate}, A::Vector{SOSASAction})
    SIndex = Dict{CASstate, Integer}()
    for (s, state) ∈ enumerate(S)
        SIndex[state] = s
    end
    AIndex = Dict{SOSASAction, Integer}()
    for (a, action) ∈ enumerate(A)
        AIndex[action] = a
    end
    return SIndex, AIndex
end

function generate_states(AV::DomainSSP, H::CASSP)
    states = Vector{CASstate}()
    G = Set{CASstate}()
    for sh in H.𝒮.F.SH
        for state in D.S
            for σ in ['⊕', '⊘', '∅']
                new_state = CASstate(sh, state, σ)
                push!(states, new_state)
                if state in D.G && σ == '∅'
                    push!(G, new_state)
                end
            end
        end
    end
    o1, o2 = rand(1:2), rand(1:2)
    oa = (o1 == 1) ? 1 : 2
    sh = [o1, o2, oa]
    return states, CASstate(sh, D.s₀, '∅'), G
end

function terminal(sosas::SOSAS, state::CASstate)
    return state in sosas.G
end

function set_route(M::SOSAS, init, goal, w)
    AV, H = M.AV, M.H
    set_route(AV, H, init, goal, w)
    generate_transitions!(H.𝒮.D, H.𝒮.A, H.𝒮.F, H, H.S, H.A, H.G)
    L1 = solve_model(AV)
    L2 = solve_model(H)
    M.s₀ = H.s₀
    M.G = H.G
    generate_costs!(M, L1, L2)
    generate_transitions!(M, M.S, M.A, AV, L1, H, L2)
end


function human_state_transition(sh, s, operator)
    o1, o2, oa = sh[1], sh[2], sh[3]

    T = Vector{Tuple{Vector, Float32}}()
    if o1 == 1 # Local operator available

        p_becomes_busy = 1.0 - (0.5)^s.w.active_avs

        if o2 == 1
            push!(T, ([2, 1, 2], p_becomes_busy * 0.75))
            push!(T, ([2, 2, 2], p_becomes_busy * 0.25))
            push!(T, ([1, 1, 1], (1.0-p_becomes_busy) * 0.75))
            push!(T, ([1, 2, 1], (1.0-p_becomes_busy) * 0.25))
        else
            push!(T, ([2, 1, 2], p_becomes_busy * 0.25))
            push!(T, ([2, 2, 2], p_becomes_busy * 0.75))
            push!(T, ([1, 1, 1], (1.0-p_becomes_busy) * 0.25))
            push!(T, ([1, 2, 1], (1.0-p_becomes_busy) * 0.75))
        end
        # if operator == 1 # AV is Operating
        #     # Local operator becomes busy (only happens if not using operator)
        #     p_becomes_busy = 1.0 - (0.5)^s.w.active_avs
        #     # Global operator takes over.
        #     if o2 == 1
        #         push!(T, ([2, 1, 2], p_becomes_busy * 0.75))
        #         push!(T, ([2, 2, 2], p_becomes_busy * 0.25))
        #         push!(T, ([1, 1, 1], (1.0-p_becomes_busy) * 0.75))
        #         push!(T, ([1, 2, 1], (1.0-p_becomes_busy) * 0.25))
        #     else
        #         push!(T, ([2, 1, 2], p_becomes_busy * 0.25))
        #         push!(T, ([2, 2, 2], p_becomes_busy * 0.75))
        #         push!(T, ([1, 1, 1], (1.0-p_becomes_busy) * 0.25))
        #         push!(T, ([1, 2, 1], (1.0-p_becomes_busy) * 0.75))
        #     end
        # else
        #     if o2 == 1
        #         push!(T, ([1, 1, 1], 0.75))
        #         push!(T, ([1, 2, 1], 0.25))
        #     else
        #         push!(T, ([1, 1, 1], 0.25))
        #         push!(T, ([1, 2, 1], 0.75))
        #     end
        # end
    else # Local operator unavailable --> state is [2, x, 2]
        p_becomes_active = (0.5)^s.w.active_avs
        if o2 == 1
            push!(T, ([1, 1, 1], p_becomes_active * 0.75))
            push!(T, ([1, 2, 1], p_becomes_active * 0.25))
            push!(T, ([2, 1, 2], (1.0 - p_becomes_active) * 0.75))
            push!(T, ([2, 2, 2], (1.0 - p_becomes_active) * 0.25))
        else
            push!(T, ([1, 1, 1], p_becomes_active * 0.25))
            push!(T, ([1, 2, 1], p_becomes_active * 0.75))
            push!(T, ([2, 1, 2], (1.0 - p_becomes_active) * 0.25))
            push!(T, ([2, 2, 2], (1.0 - p_becomes_active) * 0.75))
        end
    end
    return T
end

function generate_transitions!(SOSAS, S, A, AV, L1, H, L2)
    T = SOSAS.T
    for s=1:length(S)
        state = S[s]
        if state.state.w != SOSAS.s₀.state.w
            continue
        end
        println("here")
        T[s] = Dict{Int, Vector{Tuple{Int, Float64}}}()
        for a=1:length(A)
            T[s][a] = []
            th = TH(state.sh, state.state, A[a].operator)

            if A[a].actor == 1 #AV
                av_s = AV.SIndex[state.state]
                av_a = L1.π[av_s]
                t = AV.T[av_s][av_a]
                for i=1:length(t)
                    for j=1:length(th)
                        push!(T[s][a], (M.SIndex[SOSASstate(th[j][1], t[i][1], '⊕')],
                                                           (t[i][2] * th[j][2])))
                    end
                end
            else #Human
                human_s = H.SIndex[CASstate(state.sh, state.state, '∅')]
                human_a = L2.π[human_s]
                T[s][a] = H.T[human_s][human_a]
            end
        end
    end
end

function generate_costs(M::SOSAS, L1, L2, s, a)
    S, A, C = M.S, M.A, M.C
    state, action = S[s], A[a]
    if action.operator == 1
        av_s = M.AV.SIndex[state.state]
        if !L1.solved[av_s]
            av_a = solve(L1, M.AV, av_s)[1]
        else
            av_a = L1.π[av_s]
        end
        return M.AV.C[av_s][av_a] + autonomy_cost(state)
    else
        human_s = M.H.SIndex[CASstate(state.sh, state.state, '⊘')]
        if human_s ∉ L2.solved
            human_a = solve(L2, M.H, human_s)[1]
        else
            human_a = L2.π[human_s]
        end
        if state.σ == '⊕'
            return M.H.C[human_s][human_a] - autonomy_cost(state)
        else
            return M.H.C[human_s][human_a]
        end
    end
end

function generate_costs!(M::SOSAS, L1, L2)
    for s = 1:length(M.S)
        if M.S[s].state.w != M.s₀.state.w
            continue
        end
        for a = 1:length(M.A)
            M.C[s][a] = generate_costs(M, L1, L2, s, a)
        end
    end
end

function generate_successor(M::SOSAS,
                        state::CASstate,
                       action::CASaction)
    s, a = M.SIndex[state], M.AIndex[action]
    thresh = rand()
    p = 0.
    T = H.T[s][a]
    for (s′, prob) ∈ T
        p += prob
        if p >= thresh
            return H.S[s′]
        end
    end
end

function build_sosas(AV, L1, H, L2)
    S, s₀, G = generate_states(AV, H)
    A = [SOSASAction(1), SOSASAction(2)]
    T = Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}}()
    costs = [[0. for a=1:length(A)] for s=1:length(S)]
    M = SOSAS(AV, H, S, A, T, costs, s₀, G)
    generate_costs!(M, L1, L2)
    generate_transitions!(M, S, A, AV, L1, H, L2)
    # check_transition_validity(M)
    return M
end

function solve_model(sosas::SOSAS)
    ℒ = LRTDPsolver(sosas, 1000., 10000, .01, Dict{Int, Int}(),
                    false, Set{Int}(), zeros(length(sosas.AV.S)),
                    zeros(length(sosas.S)), zeros(length(sosas.A)))
    solve(ℒ, sosas, sosas.SIndex[sosas.s₀])
    return ℒ
end
