using POMDPs, SARSOP, POMDPSimulators
using POMDPTools
using Infiltrator
include("MDP.jl")
include("CAS.jl")

# Constants
stop_reliability = 0.85
edge_reliability = 0.75
go_reliability = 0.65
# function POPOMDP(M::DomainSSP)
#     return POPOMDP(M)
# end

## Initialize Model Parameters
W = get_random_world_state()
M = build_model(W)
𝒞 = build_cas(M, [0,1,2], ['∅', '⊘'])

# Generate states
S = 𝒞.S

# Generate actions
A = 𝒞.A

# POMDP Transition function
function POMDPs.transition(𝒫::POMDP, state::CASstate, action::CASaction)
    s, a = 𝒫.𝒞.SIndex[state], 𝒫.𝒞.AIndex[action]
    return SparseCat(S, 𝒫.𝒞.T[s][a])
end

# POMDP Reward function
function POMDPs.reward(𝒫::POMDP, state::CASstate, action::CASaction)
    s, a = 𝒫.𝒞.SIndex[state], 𝒫.𝒞.AIndex[action]
    return 𝒫.𝒞.R[s][a]
end

# POMDP Observation
struct Observation
    trailing::Bool
    oncoming::Int
end

# Generate observation set O
OIndex = Dict{Observation, Integer}()
function build_observations()
    Ω = Vector{Observation}()
    index_ = 1
    for t=0:1
        for o=-1:3
            ω = Observation(t, o)
            push!(Ω, ω)
            OIndex[ω] = index_
            index_ += 1
        end
    end
    return Ω
end
Ω = build_observations()
function index(ω::Observation)
    return OIndex[ω]
end

# Generate observation function Ω
function generate_observations()
    O = Dict{Int, Dict{Int, SparseCat}}()
    for (a, action) in enumerate(A)
        O[a] = Dict{Int, SparseCat}()
        for (sp, statePrime) in enumerate(S)
            if action.action.value == :stop
                P = Vector{Float64}()
                for ω ∈ Ω
                    p = 1.0

                    # Trailing observation
                    if ω.trailing == statePrime.state.w.trailing
                        p *= stop_reliability
                    else
                        p *= (1.0 - stop_reliability)
                    end

                    # Oncoming observation
                    if ω.oncoming == statePrime.state.oncoming == -1
                        p *= 1.0
                    elseif ω.oncoming == statePrime.state.oncoming
                        p *= stop_reliability
                    elseif ω.oncoming == -1
                        p *= (1.0 - stop_reliability)
                    else
                        p *= 0.0
                    end
                    push!(P, p)
                end
                O[a][sp] = SparseCat(Ω, P)
            elseif action.action.value == :edge
                P = Vector{Float64}()
                for ω ∈ Ω
                    p = 1.0

                    # Trailing observation
                    if ω.trailing == statePrime.state.w.trailing
                        p *= edge_reliability
                    else
                        p *= (1.0 - edge_reliability)
                    end

                    # Oncoming observation
                    if ω.oncoming == statePrime.state.oncoming == -1
                        p *= 1.0
                    elseif ω.oncoming == statePrime.state.oncoming
                        p *= edge_reliability
                    elseif ω.oncoming == -1
                        p *= (1.0 - edge_reliability)
                    else
                        p *= 0.0
                    end

                    push!(P, p)
                end
                O[a][sp] = SparseCat(Ω, P)
            else
                P = Vector{Float64}()
                for ω ∈ Ω
                    p = 1.0

                    # Trailing observation
                    if ω.trailing == statePrime.state.w.trailing
                        p *= go_reliability
                    else
                        p *= (1.0 - go_reliability)
                    end

                    # Oncoming observation
                    if ω.oncoming == statePrime.state.oncoming == -1
                        p *= 1.0
                    elseif ω.oncoming == statePrime.state.oncoming
                        p *= go_reliability
                    elseif ω.oncoming == -1
                        p *= (1.0 - go_reliability)
                    else
                        p *= 0.0
                    end

                    push!(P, p)
                end
                O[a][sp] = SparseCat(Ω, P)
            end
        end
    end
    return O
end
O = generate_observations()
function POMDPs.observation(𝒫::POMDP, action::CASaction, state::CASstate)
    a, sp = 𝒫.𝒞.AIndex[action], 𝒫.𝒞.SIndex[state]
    return 𝒫.O[a][sp]
end

struct POPOMDP <: POMDP{CASstate, CASaction, Observation}
    𝒞::CASMDP
    O::Dict{Int, Dict{Int, SparseCat}}
    OIndex::Dict{Observation, Int}
end

POMDPs.states(𝒫::POPOMDP) = S
POMDPs.actions(𝒫::POPOMDP) = A
POMDPs.actions(𝒫::POPOMDP, state) = [action for action in A if allowed(𝒫.𝒞, 𝒫.𝒞.SIndex[state], 𝒫.𝒞.AIndex[action])]
POMDPs.observations(𝒫::POPOMDP) = Ω
POMDPs.isterminal(𝒫::POPOMDP, state::CASstate) = terminal(𝒫.𝒞, state)
POMDPs.discount(𝒫::POPOMDP) = 0.9
POMDPs.initialstate(𝒫::POPOMDP) = Deterministic(𝒫.𝒞.s₀)
POMDPs.stateindex(𝒫::POPOMDP, state::CASstate) = 𝒫.𝒞.SIndex[state]
POMDPs.actionindex(𝒫::POPOMDP, action::CASaction) = 𝒫.𝒞.AIndex[action]
POMDPs.obsindex(𝒫::POPOMDP, ω::Observation) = 𝒫.OIndex[ω]

𝒫 = POPOMDP(𝒞, O, OIndex)

# =================== SOLVER CONFIGURATION ===================

@time begin
    solver = SARSOPSolver()
    policy = @time SARSOP.solve(solver, 𝒫)
end

begin
    rsum = 0.0
    rewards = Vector{Float64}()

    @time for i=1:100
        global rsum = 0.0
        for (s,b,a,o,r) in stepthrough(𝒫, policy, "s,b,a,o,r", max_steps=1000)
            println("s: $s, a: $a, o: $o, rsum: $rsum")
            r = POMDPs.reward(𝒫, s, a)
            global rsum += r

            if terminal(𝒫.𝒞, s)
                break
            end
        end
        push!(rewards, rsum)
    end

    println("Average reward: $(mean(rewards)) ± $(std(rewards))")
    println(rewards)
end
