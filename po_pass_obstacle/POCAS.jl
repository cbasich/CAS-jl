using POMDPs, QMDP, SARSOP, BasicPOMCP, ARDESPOT, POMDPSimulators
# using PointBasedValueIteration, DiscreteValueIteration
using POMDPTools, ParticleFilters
using Infiltrator

include("MDP.jl")
include("CAS.jl")

# Constants
stop_reliability = 0.95
edge_reliability = 0.85
go_reliability = 0.75

weather_reliability = Dict(
    "sunny" => 1.0,
    "rainy" => 0.8,
    "snowy" => 0.6
)

time_reliability = Dict(
    "day" => 1.0,
    "night" => 0.8
)

# stop_reliability = 1.0
# edge_reliability = 1.0
# go_reliability = 1.0
# function POPOMDP(M::DomainSSP)
#     return POPOMDP(M)
# end

## Initialize Base Models
# W = get_random_world_state()
W = WorldState(false, false, "day", "sunny")
M = build_model(W)
𝒞 = build_cas(M, [0,1,2], ['∅', '⊘'])

# Generate states
S = 𝒞.S

# Generate actions
A = 𝒞.A

## Define POMDP state

# POMDP Observation
struct Observation
    trailing_intensity::Int
    trailing_seen::Bool
    oncoming_intensity::Int
    oncoming_seen::Bool
end

struct Pstate
    s::CASstate
    ω::Observation
end

function ==(a::Pstate, b::Pstate)
    return a.s == b.s && a.ω == b.ω
end

function Base.hash(a::Pstate, h::UInt)
    h = hash(a.s, h)
    h = hash(a.ω, h)
    return h
end

struct POPOMDP <: POMDP{Pstate, CASaction, Observation}
    𝒞::CASMDP
    O::Dict{Int, Dict{Int, SparseCat}}
    OIndex::Dict{Observation, Int}
end

# Generate observation set O
OIndex = Dict{Observation, Integer}()
function build_observations()
    Ω = Vector{Observation}()
    index_ = 1
    for t_i=0:2
        for t_s in [false, true]
            for o_i=0:2
                for o_s in [false, true]
                    ω = Observation(t_i, t_s, o_i, o_s)
                    push!(Ω, ω)
                    OIndex[ω] = index_
                    index_ += 1
                end
            end
        end
    end
    return Ω
end
Ω = build_observations()
POMDPs.observations(𝒫::POPOMDP) = Ω
function index(ω::Observation)
    return OIndex[ω]
end

# Generate POMDP states
SIndex = Dict{Pstate, Integer}()
function build_states()
    PS = Vector{Pstate}()
    index_ = 1
    for s in S
        for ω in Ω
            state = Pstate(s, ω)
            push!(PS, state)
            SIndex[state] = index_
            index_ += 1
        end
    end
    return PS
end
PS = build_states()
POMDPs.states(𝒫::POPOMDP) = PS
function index(s::Pstate)
    return SIndex[s]
end

# Generate POMDP actions
POMDPs.actions(𝒫::POPOMDP) = A

# Generate observation function Ω
function generate_observations()
    O = Dict{Int, Dict{Int, SparseCat}}()
    for (a, action) in enumerate(A)
        O[a] = Dict{Int, SparseCat}()
        for (sp, statePrime) in enumerate(S)
            state = statePrime.state
            w = state.w
            P = Vector{Float64}()
            for ω ∈ Ω
                p = 1.0

                # Trailing
                if w.trailing
                    if w.weather == "sunny"
                        if w.time == "day"
                            if ω.trailing_seen
                                p *= 0.95
                            else
                                p *= 0.05
                            end
                            p *= [0.5 0.25 0.25][ω.trailing_intensity + 1]
                        else
                            if ω.trailing_seen
                                p *= 0.3
                            else
                                p *= 0.7
                            end
                            p *= [0.05 0.05 0.9][ω.trailing_intensity + 1]
                        end
                    elseif w.weather == "rainy"
                        if w.time == "day"
                            if ω.trailing_seen
                                p *= 0.75
                            else
                                p *= 0.25
                            end
                            p *= [0.2 0.3 0.5][ω.trailing_intensity + 1]
                        else
                            if ω.trailing_seen
                                p *= 0.2
                            else
                                p *= 0.8
                            end
                            p *= [0.15 0.6 0.25][ω.trailing_intensity + 1]
                        end
                    else
                        if w.time == "day"
                            if ω.trailing_seen
                                p *= 0.6
                            else
                                p *= 0.4
                            end
                            p *= [0.3 0.5 0.2][ω.trailing_intensity + 1]
                        else
                            if ω.trailing_seen
                                p *= 0.1
                            else
                                p *= 0.9
                            end
                            p *= [0.6 0.3 0.1][ω.trailing_intensity + 1]
                        end
                    end
                else
                    if ω.trailing_seen
                        p *= 0.1
                    else
                        p *= 0.9
                    end
                    p *= [0.9 0.05 0.05][ω.trailing_intensity + 1]
                end

                # Oncoming
                if state.oncoming == -1
                    if ω.oncoming_seen
                        p *= 0.0
                    end
                    p *= [1.0 0.0 0.0][ω.oncoming_intensity + 1]
                elseif state.oncoming == 0
                    if ω.oncoming_seen
                        p *= 0.1
                    else
                        p *= 0.9
                    end
                    p *= [0.9 0.05 0.05][ω.oncoming_intensity + 1]
                elseif state.oncoming == 1
                    if w.weather == "sunny"
                        if w.time == "day"
                            if ω.oncoming_seen
                                p *= 0.95
                            else
                                p *= 0.05
                            end
                            p *= [0.2 0.75 0.05][ω.oncoming_intensity + 1]
                        else
                            if ω.oncoming_seen
                                p *= 0.2
                            else
                                p *= 0.8
                            end
                            p *= [0.05 0.7 0.25][ω.oncoming_intensity + 1]
                        end
                    elseif w.weather == "rainy"
                        if w.time == "day"
                            if ω.trailing_seen
                                p *= 0.65
                            else
                                p *= 0.35
                            end
                            p *= [0.15 0.5 0.35][ω.trailing_intensity + 1]
                        else
                            if ω.trailing_seen
                                p *= 0.1
                            else
                                p *= 0.9
                            end
                            p *= [0.2 0.35 0.45][ω.trailing_intensity + 1]
                        end
                    else
                        if w.time == "day"
                            if ω.trailing_seen
                                p *= 0.6
                            else
                                p *= 0.4
                            end
                            p *= [0.6 0.3 0.1][ω.trailing_intensity + 1]
                        else
                            if ω.trailing_seen
                                p *= 0.1
                            else
                                p *= 0.9
                            end
                            p *= [0.3 0.5 0.2][ω.trailing_intensity + 1]
                        end
                    end
                else
                    if w.weather == "sunny"
                        if w.time == "day"
                            if ω.oncoming_seen
                                p *= 0.99
                            else
                                p *= 0.01
                            end
                            p *= [0.15 0.35 0.5][ω.oncoming_intensity + 1]
                        else
                            if ω.oncoming_seen
                                p *= 0.4
                            else
                                p *= 0.6
                            end
                            p *= [0.01 0.09 0.9][ω.oncoming_intensity + 1]
                        end
                    elseif w.weather == "rainy"
                        if w.time == "day"
                            if ω.trailing_seen
                                p *= 0.8
                            else
                                p *= 0.2
                            end
                            p *= [0.25 0.45 0.3][ω.trailing_intensity + 1]
                        else
                            if ω.trailing_seen
                                p *= 0.1
                            else
                                p *= 0.9
                            end
                            p *= [0.05 0.3 0.65][ω.trailing_intensity + 1]
                        end
                    else
                        if w.time == "day"
                            if ω.trailing_seen
                                p *= 0.7
                            else
                                p *= 0.3
                            end
                            p *= [0.5 0.3 0.2][ω.trailing_intensity + 1]
                        else
                            if ω.trailing_seen
                                p *= 0.05
                            else
                                p *= 0.95
                            end
                            p *= [0.1 0.6 0.3][ω.trailing_intensity + 1]
                        end
                    end
                end
                push!(P, p)
            end
            O[a][sp] = SparseCat(Ω, P)
        end
    end
    return O
end
O = generate_observations()
function POMDPs.observation(𝒫::POPOMDP, action::CASaction, state::Pstate)
    a, sp = 𝒫.𝒞.AIndex[action], 𝒫.𝒞.SIndex[state.s]
    return 𝒫.O[a][sp]
end
POMDPs.observations(𝒫::POPOMDP) = O

# POMDP Transition function
function POMDPs.transition(𝒫::POMDP, state::Pstate, action::CASaction)
    ω = state.ω
    T = zeros(length(states(𝒫)))

    if action.l > competence(ω, action.action, state.s.state)
        T[index(state)] = 1.0
        return SparseCat(states(𝒫), T)
    end

    state = state.s
    s, a = 𝒫.𝒞.SIndex[state], 𝒫.𝒞.AIndex[action]
    P = 𝒫.𝒞.T[s][a]
    for (sp, stateprime) in enumerate(PS)
        ωprime = stateprime.ω
        cstateprime = stateprime.s
        csp = 𝒫.𝒞.SIndex[cstateprime]
        p = P[csp]

        ω_P = last.(O[a][csp])
        p *= ω_P[obsindex(𝒫, ω)]

        T[sp] = p
    end
    return SparseCat(states(𝒫), T)
    # s, a = 𝒫.𝒞.SIndex[state], 𝒫.𝒞.AIndex[action]
    # return SparseCat(S, 𝒫.𝒞.T[s][a])
end

# POMDP Reward function
function POMDPs.reward(𝒫::POMDP, state::Pstate, action::CASaction)
    s, a = 𝒫.𝒞.SIndex[state.s], 𝒫.𝒞.AIndex[action]
    # if !allowed(𝒫.𝒞, s, a)
    #     return -1000.0
    # end
    return 𝒫.𝒞.R[s][a]
end

POMDPs.isterminal(𝒫::POPOMDP, state::Pstate) = terminal(𝒫.𝒞, state.s.state)
POMDPs.initialstate(𝒫::POPOMDP) = Deterministic(Pstate(𝒫.𝒞.s₀, Observation(0, false, 0, false)))
POMDPs.initialobs(𝒫::POPOMDP) = Observation(0, false, 0, false)
POMDPs.stateindex(𝒫::POPOMDP, state::Pstate) = index(state)
POMDPs.actionindex(𝒫::POPOMDP, action::CASaction) = 𝒫.𝒞.AIndex[action]
POMDPs.obsindex(𝒫::POPOMDP, ω::Observation) = 𝒫.OIndex[ω]
POMDPs.discount(𝒫::POPOMDP) = 0.9

function competence(ω::Observation, action::DomainAction, state::DomainState)
    time = state.w.time
    weather = state.w.weather

    if weather == "snowy" && time == "night"
        return 0
    end

    if state.position == 4
        return 2
    end

    if action.value == :stop
        if state.position > 1 && (ω.oncoming_seen || ω.oncoming_intensity > 0)
            return 0
        elseif (state.position == 1 && state.w.waiting &&
               (!ω.oncoming_seen && ω.oncoming_intensity < 2) &&
               (ω.trailing_seen || ω.trailing_intensity == 2))
            return 0
        else
            return 2
        end
    elseif action.value == :edge
        if state.position > 0
            return 0
        else
            return 2
        end
    else
        if (state.position == 1 && (
             (weather == "sunny" && (!ω.oncoming_seen ||
              ω.oncoming_intensity == 0)) ||
             (weather == "rainy" && time == "day" &&
              !ω.oncoming_seen && ω.oncoming_intensity < 2) ||
             (!ω.oncoming_seen && ω.oncoming_intensity == 0)))
            return 2
        elseif (state.position == 2 && (
             ((weather == "sunny" || (weather == "rainy" && time == "day")) &&
             (!ω.oncoming_seen || ω.oncoming_intensity < 2)) ||
             (weather == "snowy" && time == "day" && !ω.oncoming_seen &&
              ω.oncoming_intensity < 2) ||
             (!ω.oncoming_seen && ω.oncoming_intensity == 0)))
            return 2
        elseif state.position == 3
            return 2
        else
            return 0
        end
    end
end

function competence(ω::Observation, action::DomainAction, U::Array{Pstate,1})
    return competence(ω, action, U[1].s.state)
end


# =================== SOLVER CONFIGURATION ===================
𝒫 = POPOMDP(𝒞, O, OIndex)
@time begin
    ### SARSOP
    # solver = SARSOPSolver()
    # policy = @time SARSOP.solve(solver, 𝒫)

    ### QMDP
    # solver = QMDPSolver(SparseValueIterationSolver(max_iterations=1000, belres=1e-3, verbose=true))
    # policy = SARSOP.solve(solver, 𝒫)

    ### PBVI
    # solver = PBVISolver(verbose=true)
    # policy = PointBasedValueIteration.solve(solver, 𝒫)

    ### POMCP
    # solver = POMCPSolver()
    # planner = BasicPOMCP.solve(solver, 𝒫)

    ### ARDESPOT
    solver = DESPOTSolver(bounds = IndependentBounds(-30.0, 0.0,
        check_terminal=true), default_action=CASaction(DomainAction(:go), 0))
    planner = ARDESPOT.solve(solver, 𝒫)
end

begin
    rsum = 0.0
    rewards = Vector{Float64}()
    println("  ")
    @time for i=1:1
        global rsum = 0.0

        # W = get_random_world_state()
        # W = WorldState(false, false, "day", "sunny")
        # M.s₀ = DomainState(M.s₀.position, M.s₀.oncoming, M.s₀.priority, W)
        # 𝒞.s₀ = CASstate(M.s₀, '∅')

        ### OFFLINE ###
        # for (s,b,a,o,r) in stepthrough(𝒫, policy, "s,b,a,o,r", max_steps=1000)
        #     r = POMDPs.reward(𝒫, s, a)
        #     global rsum += r
        #     # println("s: $s, a: $a, o: $o, r: $r, rsum: $rsum")
        #     # if a ∉ POMDPs.actions(𝒫, o)
        #         # print("Bad action.")
        #     # end
        #     if terminal(𝒫.𝒞, s)
        #         break
        #     end
        # end
        # println("Simulating")
        ### ONLINE ###
        filter = BootstrapFilter(𝒫, 10)
        for (s,a,o,b,r,) in stepthrough(𝒫, planner, filter, "s,a,o,b,r", max_steps=100)
            # println("Step...")
            # r = POMDPs.reward(𝒫, s, a)
            global rsum += r
            println("s: $s, a: $a, o: $o, r: $r, rsum: $rsum, AI: $action_info")
            @infiltrate
            if a.l > competence(o, a.action, unique(collect(particles(b))))
                @infiltrate
                print("Bad action.")
            end


            if POMDPs.isterminal(𝒫, s)
                # println("Terminating")
                # println(s)
                break
            end
        end
        push!(rewards, rsum)
    end

    println("Average reward: $(mean(rewards)) ± $(std(rewards))")
    # println(rewards)
end
