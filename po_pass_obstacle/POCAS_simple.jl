using POMDPs, QMDP, SARSOP, BasicPOMCP, ARDESPOT, POMDPSimulators
using PointBasedValueIteration, DiscreteValueIteration
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
    # trailing_intensity::Int
    trailing_seen::Bool
    # oncoming_intensity::Int
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

# Generate observation set Ω
OIndex = Dict{Observation, Integer}()
function build_observations()
    Ω = Vector{Observation}()
    index_ = 1
    # for t_i=0:2
    #     for t_s in [false, true]
    #         for o_i=0:2
    #             for o_s in [false, true]
    #                 ω = Observation(t_i, t_s, o_i, o_s)
    #                 push!(Ω, ω)
    #                 OIndex[ω] = index_
    #                 index_ += 1
    #             end
    #         end
    #     end
    # end
    for t_s in [false, true]
        for o_s in [false, true]
            ω = Observation(t_s, o_s)
            push!(Ω, ω)
            OIndex[ω] = index_
            index_ += 1
        end
    end
    return Ω
end
Ω = build_observations()
POMDPs.observations(𝒫::POPOMDP) = Ω
POMDPs.obsindex(𝒫::POPOMDP, ω::Observation) = 𝒫.OIndex[ω]
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

# Generate observation function O
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
                            # p *= [0.5 0.25 0.25][ω.trailing_intensity + 1]
                        else
                            if ω.trailing_seen
                                p *= 0.3
                            else
                                p *= 0.7
                            end
                            # p *= [0.05 0.05 0.9][ω.trailing_intensity + 1]
                        end
                    elseif w.weather == "rainy"
                        if w.time == "day"
                            if ω.trailing_seen
                                p *= 0.75
                            else
                                p *= 0.25
                            end
                            # p *= [0.2 0.3 0.5][ω.trailing_intensity + 1]
                        else
                            if ω.trailing_seen
                                p *= 0.2
                            else
                                p *= 0.8
                            end
                            # p *= [0.15 0.6 0.25][ω.trailing_intensity + 1]
                        end
                    else
                        if w.time == "day"
                            if ω.trailing_seen
                                p *= 0.6
                            else
                                p *= 0.4
                            end
                            # p *= [0.3 0.5 0.2][ω.trailing_intensity + 1]
                        else
                            if ω.trailing_seen
                                p *= 0.1
                            else
                                p *= 0.9
                            end
                            # p *= [0.6 0.3 0.1][ω.trailing_intensity + 1]
                        end
                    end
                else
                    if ω.trailing_seen
                        p *= 0.1
                    else
                        p *= 0.9
                    end
                    # p *= [0.9 0.05 0.05][ω.trailing_intensity + 1]
                end

                # Oncoming
                if state.oncoming == -1
                    if ω.oncoming_seen
                        p *= 0.0
                    end
                    # p *= [1.0 0.0 0.0][ω.oncoming_intensity + 1]
                elseif state.oncoming == 0
                    if ω.oncoming_seen
                        p *= 0.0
                    else
                        p *= 1.0
                    end
                    # p *= [0.9 0.05 0.05][ω.oncoming_intensity + 1]
                elseif state.oncoming == 1
                    if w.weather == "sunny"
                        if w.time == "day"
                            if ω.oncoming_seen
                                p *= 0.95
                            else
                                p *= 0.05
                            end
                            # p *= [0.2 0.75 0.05][ω.oncoming_intensity + 1]
                        else
                            if ω.oncoming_seen
                                p *= 0.2
                            else
                                p *= 0.8
                            end
                            # p *= [0.05 0.7 0.25][ω.oncoming_intensity + 1]
                        end
                    elseif w.weather == "rainy"
                        if w.time == "day"
                            if ω.oncoming_seen
                                p *= 0.65
                            else
                                p *= 0.35
                            end
                            # p *= [0.15 0.5 0.35][ω.oncoming_intensity + 1]
                        else
                            if ω.oncoming_seen
                                p *= 0.1
                            else
                                p *= 0.9
                            end
                            # p *= [0.2 0.35 0.45][ω.oncoming_intensity + 1]
                        end
                    else
                        if w.time == "day"
                            if ω.oncoming_seen
                                p *= 0.6
                            else
                                p *= 0.4
                            end
                            # p *= [0.6 0.3 0.1][ω.oncoming_intensity + 1]
                        else
                            if ω.oncoming_seen
                                p *= 0.1
                            else
                                p *= 0.9
                            end
                            # p *= [0.3 0.5 0.2][ω.oncoming_intensity + 1]
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
                            # p *= [0.15 0.35 0.5][ω.oncoming_intensity + 1]
                        else
                            if ω.oncoming_seen
                                p *= 0.4
                            else
                                p *= 0.6
                            end
                            # p *= [0.01 0.09 0.9][ω.oncoming_intensity + 1]
                        end
                    elseif w.weather == "rainy"
                        if w.time == "day"
                            if ω.oncoming_seen
                                p *= 0.8
                            else
                                p *= 0.2
                            end
                            # p *= [0.25 0.45 0.3][ω.oncoming_intensity + 1]
                        else
                            if ω.oncoming_seen
                                p *= 0.1
                            else
                                p *= 0.9
                            end
                            # p *= [0.05 0.3 0.65][ω.oncoming_intensity + 1]
                        end
                    else
                        if w.time == "day"
                            if ω.oncoming_seen
                                p *= 0.7
                            else
                                p *= 0.3
                            end
                            # p *= [0.5 0.3 0.2][ω.oncoming_intensity + 1]
                        else
                            if ω.oncoming_seen
                                p *= 0.05
                            else
                                p *= 0.95
                            end
                            # p *= [0.1 0.6 0.3][ω.oncoming_intensity + 1]
                        end
                    end
                end
                push!(P, p)
            end
            if round(sum(P)) != 1.0
                @infiltrate
            end
            O[a][sp] = SparseCat(Ω, P)
        end
    end
    return O
end
O = generate_observations()
function POMDPs.observation(𝒫::POPOMDP, action::CASaction, state::Pstate)
    ωp = obsindex(𝒫, state.ω)
    P = zeros(length(Ω))
    P[ωp] = 1.0
    return SparseCat(Ω, P)
end

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
        if state.position > 1 && (ω.oncoming_seen) # || ω.oncoming_intensity > 0)
            return 0
        elseif (state.position == 1 && state.w.waiting &&
                !ω.oncoming_seen && ω.trailing_seen)
            return 1
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
        if (state.position == 1 && ( #state.priority ||
           (weather == "sunny" && !ω.oncoming_seen) ||
           (weather == "rainy" && time == "day" && !ω.oncoming_seen)))
            return 2
        elseif (state.position == 2 && ( #state.priority ||
               (weather == "sunny" || time == "day") && !ω.oncoming_seen))
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

# POMDP Transition function
function POMDPs.transition(𝒫::POMDP, state::Pstate, action::CASaction)
    ω = state.ω
    state = state.s
    T = zeros(length(states(𝒫)))

    s, a = 𝒫.𝒞.SIndex[state], 𝒫.𝒞.AIndex[action]
    P = 𝒫.𝒞.T[s][a]
    for (sp, stateprime) in enumerate(PS)
        ωprime = stateprime.ω
        ωp = obsindex(𝒫, ωprime)
        cstateprime = stateprime.s
        csp = 𝒫.𝒞.SIndex[cstateprime]
        p = P[csp]

        ω_P = O[a][csp].probs
        p *= ω_P[ωp]

        T[sp] = p
    end
    return SparseCat(states(𝒫), T)
end

# POMDP Reward function
function POMDPs.reward(𝒫::POMDP, state::Pstate, action::CASaction)
    s, a = 𝒫.𝒞.SIndex[state.s], 𝒫.𝒞.AIndex[action]
    if action.l > competence(state.ω, action.action, state.s.state)
        return -1000.0
    end
    return 𝒫.𝒞.R[s][a]
end


POMDPs.actions(𝒫::POPOMDP, b) = [a for a in A if a.l <= competence(unique(
        collect(particles(b)))[1].ω, a.action, unique(collect(particles(b))))]
POMDPs.isterminal(𝒫::POPOMDP, state::Pstate) = terminal(𝒫.𝒞, state.s)
POMDPs.initialstate(𝒫::POPOMDP) = Deterministic(Pstate(𝒫.𝒞.s₀, Observation(false, false)))
POMDPs.initialobs(𝒫::POPOMDP) = Observation(false, false)
POMDPs.stateindex(𝒫::POPOMDP, state::Pstate) = index(state)
POMDPs.actionindex(𝒫::POPOMDP, action::CASaction) = 𝒫.𝒞.AIndex[action]
POMDPs.discount(𝒫::POPOMDP) = 0.9
𝒫 = POPOMDP(𝒞, O, OIndex)


################################
# Learning and Data Management #
################################
function get_features(state::Pstate)
    dstate = state.s.state
    ω = state.ω
    x = [dstate.position, dstate.w.time, dstate.w.weather, dstate.w.waiting,
         ω.oncoming_seen, ω.trailing_seen]
end

function update_feedback_profile!(𝒫::POPOMDP)
    for (a, action) in enumerate(A):
        X, Y, RF = nothing, nothing, nothing
        try
            X, Y = split_data()
            RF = build_forest(Y, X, -1, 11, 0.7, -1)
        catch
            continue
        end
        for (s, state) in enumerate(states(𝒫))
            f = get_features(state)
            pred = apply_forest_proba(RF, f, [0,1])
            λ[s][a]
end

# for s in states(𝒫)
#     for a in A
#         T = transition(𝒫, s, a)
#         if round(sum(T.probs), digits=5) != 1.0
#             print(s, " | ", a)
#             @infiltrate
#         end
#     end
# end
#
# for a in A
#     for s in states(𝒫)
#         if round(sum(observation(𝒫, a, s).probs), digits=5) != 1.0
#             print(a, "   |   ", s)
#         end
#     end
# end
# struct thisPOMDPSimIterator{SPEC, M<:POMDP, P<:Policy, U<:Updater, RNG<:AbstractRNG, B, S}
#     pomdp::M
#     policy::P
#     updater::U
#     rng::RNG
#     init_belief::B
#     init_state::S
#     max_steps::Int
# end
# function POMDPTools.Simulators.POMDPSimIterator(spec::Union{Tuple,Symbol}, pomdp::POMDP, policy::Policy, up::Updater, rng::AbstractRNG, init_belief, init_state, max_steps::Int)
#     return thisPOMDPSimIterator{spec,
#                             typeof(pomdp),
#                             typeof(policy),
#                             typeof(up),
#                             typeof(rng),
#                             typeof(init_belief),
#                             typeof(init_state)}(pomdp,
#                                                 policy,
#                                                 up,
#                                                 rng,
#                                                 init_belief,
#                                                 init_state,
#                                                 max_steps)
# end
# function Base.iterate(it::thisPOMDPSimIterator, is::Tuple{Int,S,B} = (1, it.init_state, it.init_belief)) where {S,B}
#     if isterminal(it.pomdp, is[2]) || is[1] > it.max_steps
#         return nothing
#     end
#     t = is[1]
#     s = is[2]
#     b = is[3]
#     a, ai = action_info(it.policy, b)
#     out = @gen(:sp,:o,:r,:info)(it.pomdp, s, a, it.rng)
#     outnt = NamedTuple{(:sp,:o,:r,:info)}(out)
#     bp, ui = update_info(it.updater, b, a, outnt.o)
#     nt = merge(outnt, (t=t, b=b, s=s, a=a, action_info=ai, bp=bp, update_info=ui))
#     @infiltrate
#     return (out_tuple(it, nt), (t+1, nt.sp, nt.bp))
# end
#
# function ParticleFilters.update(up::BasicParticleFilter, b::ParticleCollection, a, o)
#     pm = up._particle_memory
#     wm = up._weight_memory
#     resize!(pm, n_particles(b))
#     resize!(wm, n_particles(b))
#     predict!(pm, up.predict_model, b, a, o, up.rng)
#     reweight!(wm, up.reweight_model, b, a, pm, o, up.rng)
#
#     return resample(up.resampler,
#                     WeightedParticleBelief(pm, wm, sum(wm), nothing),
#                     up.predict_model,
#                     up.reweight_model,
#                     b, a, o,
#                     up.rng)
# end

# =================== SOLVER CONFIGURATION ===================
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
    # solver = POMCPSolver(default_action=CASaction(DomainAction(:go), 0))
    # planner = BasicPOMCP.solve(solver, 𝒫)

    ### ARDESPOT
    solver = DESPOTSolver(T_max = 10, bounds = IndependentBounds(-50.0, 0.0,
        check_terminal=true), default_action=CASaction(DomainAction(:go), 0))
    planner = ARDESPOT.solve(solver, 𝒫)
end

begin
    rsum = 0.0
    rewards = Vector{Float64}()
    println("  ")
    @time for i=1:10
        global rsum = 0.0

        # W = get_random_world_state()
        # W = WorldState(false, false, "day", "sunny")
        # M.s₀ = DomainState(M.s₀.position, M.s₀.oncoming, M.s₀.priority, W)
        # 𝒞.s₀ = CASstate(M.s₀, '∅')

        ### OFFLINE ###
        # for (s,b,a,o,r) in stepthrough(𝒫, policy, "s,b,a,o,r", max_steps=1000)
        #     r = POMDPs.reward(𝒫, s, a)
        #     global rsum += r
        #     println("s: $s, a: $a, o: $o, r: $r, rsum: $rsum")
        #     # if a ∉ POMDPs.actions(𝒫, o)
        #         # print("Bad action.")
        #     # end
        #     # @infiltrate
        #     if a.l > competence(o, a.action, b.state_list[findall(b.b .> 0)[1]].s.state)
        #         print("Bad action.\n")
        #         @infiltrate
        #     end
        #     if isterminal(𝒫, s)
        #         break
        #     end
        # end
        # println("Simulating")

        ### ONLINE ###
        slast = initialstate(𝒫)
        filter_ = BootstrapFilter(𝒫, 10)
        for (s,a,o,b,bp,sp) in stepthrough(𝒫, planner, filter_, "s,a,o,b,bp,sp", max_steps=1000)
            println("Step...")
            println(slast == s)
            r = POMDPs.reward(𝒫, s, a)
            global rsum += r
            println(" s: $s\n a: $a\n o: $o\n r: $r, rsum: $rsum\n")
            sp = Pstate(sp.s, o)
            slast = sp
            for (i, particle) in enumerate(filter_._particle_memory)
                if particle.ω != o
                    filter_._weight_memory[i] *= 0.0
                end
            end
            if sp.ω != o
                print("error.")
            end
            if a.l > competence(s.ω, a.action, unique(collect(particles(b))))
                print("Bad action.\n")
                @infiltrate
            end
            if POMDPs.isterminal(𝒫, sp)
                println("Terminating")
                println(sp)
            end
        end
        push!(rewards, rsum)
    end
    println("Average reward: $(mean(rewards)) ± $(std(rewards))")
end
