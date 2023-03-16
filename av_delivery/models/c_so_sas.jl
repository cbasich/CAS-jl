import Base.==
include("../scripts/maps.jl")
include("../scripts/utils.jl")

struct SOSASAction
    operator::Int
end

mutable struct SOSAS
    AV
    L1
    H
    L2
    S
    A
    T
    C
    s₀
    G
    SIndex
    AIndex
end
function SOSAS(AV, L1, H, L2, S, A, T, C, s₀, G)
    SIndex, AIndex = generate_index_dicts(S, A)
    return SOSAS(AV, L1, H, L2, S, A, T, C, s₀, G, SIndex, AIndex)
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

function generate_states(AV::AVSSP, H::CASSP)
    states = Vector{CASstate}()
    G = Set{CASstate}()
    for sh in H.𝒮.F.SH
        for state in AV.S
            for σ in ['⊕', '⊘', '∅']
                new_state = CASstate(sh, state, σ)
                push!(states, new_state)
                if state in AV.G
                    push!(G, new_state)
                end
            end
        end
    end
    o1, o2 = rand(1:2), rand(1:2)
    oa = (o1 == 1) ? 1 : 2
    sh = [o1, o2, oa]
    return states, CASstate(sh, AV.s₀, '∅'), G
end

function generate_goals(AV::AVSSP, H::CASSP)
    G = Set{CASstate}()
    for sh in H.𝒮.F.SH
        for state in AV.G
            for σ in ['⊕', '⊘', '∅']
                new_state = CASstate(sh, state, σ)
                push!(G, new_state)
            end
        end
    end
    return G
end

function terminal(sosas::SOSAS, state::CASstate)
    return state.state in sosas.AV.G
end

function set_route(M::SOSAS, init, goal, w)
    AV, H = M.AV, M.H
    set_route(H.𝒮.D, H, init, goal, w)
    generate_transitions!(H.𝒮.D, H.𝒮.A, H.𝒮.F, H, H.S, H.A, H.G)
    M.L2 = solve_model(H)
    set_init!(AV, init, w)
    set_goals!(AV, [goal], w)
    generate_transitions!(AV, AV.graph)
    generate_costs!(AV)
    M.L1 = solve_model(AV)
    M.s₀ = H.s₀
    M.G = generate_goals(AV, H)
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
        if state.state.w.time != SOSAS.s₀.state.w.time || state.state.w.weather != SOSAS.s₀.state.w.weather
            continue
        end

        if (state.sh[1] == 1 && state.sh[3] == 2) || (state.sh[1] == 2 && state.sh[3] == 1)
            continue
        end

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
                        push!(T[s][a], (M.SIndex[CASstate(th[j][1], t[i][1], '⊕')],
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

function get_transition(M::SOSAS, s, a)
    AV, L1, H, L2, S, A = M.AV, M.L1, M.H, M.L2, M.S, M.A
    T = Vector{Tuple{Int, Float64}}()
    state = S[s]

    if terminal(M, state)
        return [(s, 1.0)]
    end

    th = human_state_transition(state.sh, state.state, A[a].operator)

    if A[a].operator == 1 #AV
        av_s = AV.SIndex[state.state]
        if !L1.solved[av_s]
            av_a = solve(L1, M.AV, av_s)[1]
        else
            av_a = L1.π[av_s]
        end
        if competence(state.state, AV.A[av_a]) == 0 || (typeof(state.state) == EdgeState && state.state.r != "None")
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
            for j=1:length(th)
                state′ = CASstate(th[j][1], dstate′, '⊘')
                push!(T, (M.SIndex[state′], th[j][2]))
            end
            return [(s, 1.0)]
        else
            t = AV.T[av_s][av_a]
            for i=1:length(t)
                for j=1:length(th)
                    push!(T, (M.SIndex[CASstate(th[j][1], AV.S[t[i][1]], '⊕')],
                                                          (t[i][2] * th[j][2])))
                end
            end
        end
    else #Human
        human_s = H.SIndex[CASstate(state.sh, state.state, '⊘')]
        if human_s ∉ L2.solved
            human_a = solve(L2, H, human_s)[1]
        else
            human_a = L2.π[human_s]
        end
        T = H.T[human_s][human_a]
    end

    return T
end

function generate_costs(M::SOSAS, L1, L2, s, a)
    S, A, C = M.S, M.A, M.C
    state, action = S[s], A[a]
    if terminal(M, state)
        return 0.
    end
    if action.operator == 1
        av_s = M.AV.SIndex[state.state]
        if !L1.solved[av_s]
            av_a = solve(L1, M.AV, av_s)[1]
        else
            av_a = L1.π[av_s]
        end
        return M.AV.C[av_s][av_a] + autonomy_cost(state)
    else
        human_state = CASstate(state.sh, state.state, '⊘')
        human_s = M.H.SIndex[human_state]
        if human_s ∉ L2.solved
            human_a = solve(L2, M.H, human_s)[1]
        else
            human_a = L2.π[human_s]
        end
        if state.σ == '⊕'
            return M.H.C[human_s][human_a] - autonomy_cost(human_state)
        else
            return M.H.C[human_s][human_a]
        end
    end

end

function generate_costs!(M::SOSAS, L1, L2)
    for s = 1:length(M.S)
        state = M.S[s]
        if state.state.w.time != M.s₀.state.w.time || state.state.w.weather != M.s₀.state.w.weather
            continue
        end

        if (state.sh[1] == 1 && state.sh[3] == 2) || (state.sh[1] == 2 && state.sh[3] == 1)
            continue
        end
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
                       action::SOSASAction)
    s, a = M.SIndex[state], M.AIndex[action]
    thresh = rand()
    p = 0.
    T = get_transition(M, s, a)
    for (s′, prob) ∈ T
        p += prob
        if p >= thresh
            return M.S[s′]
        end
    end
end

function build_sosas(AV, L1, H, L2)
    S, s₀, G = generate_states(AV, H)
    A = [SOSASAction(1), SOSASAction(2)]
    T = Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}}()
    costs = [[0. for a=1:length(A)] for s=1:length(S)]
    M = SOSAS(AV, L1, H, L2, S, A, T, generate_costs, s₀, G)
    return M
end

function solve_model(M::SOSAS)
    ℒ = LRTDPsolverSOSAS(M, 1000., 10000, .01, Dict{Int, Int}(),
                    false, Set{Int}(), zeros(length(M.AV.S)),
                    zeros(length(M.S)), zeros(length(M.A)))
    println("Solving SOSAS")
    solve(ℒ, M, M.SIndex[M.s₀])
    return ℒ
end
