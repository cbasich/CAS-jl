import Combinatorics

# include("utils.jl")
include("domain_model.jl")
include("../LAOStarSolver.jl")

struct CASstate
    state::DomainState
        σ::Char
end

struct CASaction
    action::DomainAction
         l::Int
end

##

struct AutonomyModel
    L::Vector{Int}
    κ::Dict{Int, Dict{Int, Vector{Int}}}
    μ
end

function generate_autonomy_profile(𝒟::DomainSSP,
                                   L::Vector{Int})
    κ = Dict{Int, Dict{Int, Vector{Int}}}()
    for (s, state) in enumerate(𝒟.S)
        κ[s] = Dict{Int, Vector{Int}}()
        for (a, action) in enumerate(𝒟.A)
            if typeof(state) == NodeState
                if state.p == true || state.o == true || state.v > 1
                    κ[s][a] = [0,1]
                else
                    κ[s][a] = L
                end
            else
                if state.o == true
                    κ[s][a] = [0,1]
                else
                    κ[s][a] = L
                end
            end
        end
    end
    return κ
end

function autonomy_cost(state::CASstate,
                      action::CASaction)
    return 1.0                          #TODO: Replace this wih the correct cost.
end
##


##
struct FeedbackModel
    Σ::Vector{Char}
    λ::Dict{Int, Dict{Int, Dict{Int, Dict{Char, Float64}}}}
    ρ::Function
end

function generate_feedback_profile(𝒟::DomainSSP,
                                   Σ::Vector{Char},
                                   L::Vector{Int})
    λ = Dict{Int, Dict{Int, Dict{Int, Dict{Char, Float64}}}}()
    for (s, state) in enumerate(𝒟.S)
        λ[s] = Dict{Int, Dict{Int, Dict{Char, Float64}}}()
        for (a, action) in enumerate(𝒟.A)
            λ[s][a] = Dict{Int, Dict{Char, Float64}}()
            for l in L
                λ[s][a][l] = Dict{Char, Float64}()
                for σ ∈ Σ
                    λ[s][a][l][σ] = get_feedback_probability(state,action,l,σ)
                end
            end
        end
    end
    return λ
end

function get_feedback_probability(state::DomainState,
                                 action::DomainAction,
                                      l::Int,
                                      σ::Char)
    # F = get_feature_vector(state)
    if l == 1 && σ == '⊕'
        return 1.0
    elseif l == 2 && σ == '∅'
        return 1.0
    else
        return 0.0
    end
end

function human_cost(state::CASstate,
                   action::CASaction)
    return 1.0                            #TODO: Replace this with correct cost.
end
##

struct CAS
    D::DomainSSP
    A::AutonomyModel
    F::FeedbackModel
end

struct CASSP
    𝒮::CAS
    S::Vector{CASstate}
    A::Vector{CASaction}
    T
    C
   s₀::CASstate
    G::Set{CASstate}
    SIndex::Dict{CASstate, Int}
    AIndex::Dict{CASaction, Int}
end
function CASSP(𝒮::CAS,
               S::Vector{CASstate},
               A::Vector{CASaction},
               T::Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}},
               C::Function,
              s₀::CASstate,
               G::Set{CASstate})
    SIndex, AIndex = generate_index_dicts(S, A)
    return CASSP(𝒮, S, A, T, C, s₀, G, SIndex, AIndex)
end

function generate_index_dicts(S::Vector{CASstate}, A::Vector{CASaction})
    SIndex = Dict{CASstate, Integer}()
    for (s, state) ∈ enumerate(S)
        SIndex[state] = s
    end
    AIndex = Dict{CASaction, Integer}()
    for (a, action) ∈ enumerate(A)
        AIndex[action] = a
    end
    return SIndex, AIndex
end

function generate_states(D, F)
    states = Vector{CASstate}()
    G = Set{CASstate}()
    for state in D.S
        for σ in F.Σ
            new_state = CASstate(state, σ)
            push!(states, new_state)
            if state in D.G
                push!(G, new_state)
            end
        end
    end
    return states, CASstate(D.s₀, '∅'), G
end

function terminal(C::CASSP, state::CASstate)
    return state in C.G
end

function generate_actions(D, A)
    actions = Vector{CASaction}()
    for action in D.A
        for l in A.L
            new_action = CASaction(action, l)
            push!(actions, new_action)
        end
    end
    return actions
end

function generate_transitions(𝒟, 𝒜, ℱ,
                              S::Vector{CASstate},
                              A::Vector{CASaction},
                              G::Set{CASstate})

    T = Dict{Int, Dict{Int, Vector{Tuple{Int, Float64}}}}()
    for (s, state) in enumerate(S)
        T[s] = Dict{Int, Vector{Tuple{Int, Float64}}}()
        for (a, action) in enumerate(A)
            if state in G
                T[s][a] = [(s, 1.0)]
            end
            T[s][a] = Vector{Tuple{Int, Float64}}()
            base_state = state.state
            base_action = action.action
            base_s = 𝒟.SIndex[base_state]
            base_a = 𝒟.AIndex[base_action]

            t = 𝒟.T[base_s][base_a]

            if action.l == 0
                # T[s][a] = transfer_control(𝒟, S, A, state, action)
                for (sp, p) in t
                    push!(T[s][a], ((sp-1) * 4 + 4, p))
                end
            elseif action.l == 1
                p_approve = ℱ.λ[base_s][base_a][action.l]['⊕']
                p_disapprove = ℱ.λ[base_s][base_a][action.l]['⊖']
                push!(T[s][a], ((base_s-1) * 4 + 2, p_disapprove))
                for (sp, p) in t
                    push!(T[s][a], ((sp-1) * 4 + 1, p * p_approve))
                end
            elseif action.l == 2
                p_override = ℱ.λ[base_s][base_a][action.l]['⊘']
                p_null = ℱ.λ[base_s][base_a][action.l]['∅']
                push!(T[s][a], ((base_s-1) * 4 + 3, p_override))
                for (sp, p) in t
                    push!(T[s][a], ((sp-1) * 4 + 4, p * p_null))
                end
            else
                for (sp, p) in t
                    push!(T[s][a], ((sp-1) * 4 + 4, p))
                end
            end
        end
    end

    return T
end

function generate_cas_probability(state::DomainState,
                                 action::DomainAction,
                                      σ::Char)
    if l == 0
        return 1.0
    elseif l == 1 && (σ == '⊘' || σ == '∅')
        return 0.0
    elseif l == 2 && (σ == '⊕' || σ == '⊖')
        return 0.0
    else

    end
    λ = ℱ.λ
    p = λ(state, action, σ)

end

function generate_costs(C::CASSP,
                        s::Int,
                        a::Int,)
    D, A, F = C.𝒮.D, C.𝒮.A, C.𝒮.F
    state, action = C.S[s], C.A[a]
    cost = D.C(D, D.SIndex[state.state], D.AIndex[action.action])
    cost += A.μ(state, action)
    cost += F.ρ(state, action)
    return cost
end

function build_cas(𝒟::DomainSSP,
                   L::Vector{Int},
                   Σ::Vector{Char})
    κ = generate_autonomy_profile(𝒟, L)
    𝒜 = AutonomyModel(L, κ, autonomy_cost)
    λ = generate_feedback_profile(𝒟, Σ, L)
    ℱ = FeedbackModel(Σ, λ, human_cost)
    𝒮 = CAS(𝒟, 𝒜, ℱ)
    S, s₀, G = generate_states(𝒟, ℱ)
    A = generate_actions(𝒟, 𝒜)
    T = generate_transitions(𝒟, 𝒜, ℱ, S, A, G)

    𝒮 = CASSP(𝒮, S, A, T, generate_costs, s₀, G)
end

function solve_model(C::CASSP)
    ℒ = LAOStarSolver(100000, 1000., 1.0, .001, Dict{Integer, Integer}(),
                        zeros(length(C.S)), zeros(length(C.S)),
                        zeros(length(C.S)), zeros(length(C.A)))
    a, total_expanded = solve(ℒ, C, C.SIndex[C.s₀])
    println("LAO* expanded $total_expanded nodes.")
    println("Expected cost to goal: $(ℒ.V[C.SIndex[C.s₀]])")
    return ℒ
end

function main()
    M = build_model()
    C = build_cas(M, [0,1,2,3], ['⊕', '⊖', '⊘', '∅'])
    ℒ = solve_model(C)
end

main()
