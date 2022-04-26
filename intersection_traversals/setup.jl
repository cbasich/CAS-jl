include("domain_model.jl")
include("cas.jl")

setup_parameters = Dict(
    "go straight" => Dict(
        :p_ped => 0.5,
        :p_occ => 0.7,
        :p_row => 0.25,
        :p_fol => 0.2,
        :p_tra => [.25 .25 .25 .25]
    ),
    "turn left" => Dict(
        :p_ped => 0.5,
        :p_occ => 0.7,
        :p_row => 0.25,
        :p_fol => 0.2,
        :p_tra => [.25 .25 .25 .25]
    ),
    "turn right" => Dict(
        :p_ped => 0.3,
        :p_occ => 0.3,
        :p_row => 0.5,
        :p_fol => 0.2,
        :p_tra => [.25 .25 .25 .25]
    ),
    "u turn" => Dict(
        :p_ped => 0.7,
        :p_occ => 0.7,
        :p_row => 0.1,
        :p_fol => 0.2,
        :p_tra => [.25 .25 .25 .25]
    )

)


function draw_start_state(maneuver)
    params = setup_parameters[maneuver]
    p = rand() <= params[:p_ped]
    o = rand() <= params[:p_occ]
    r = rand() <= params[:p_row]
    f = rand() <= params[:p_fol]
    t = rand() <= params[:p_tra]

    return State(:approaching, p, o, r, f, t)
end


function setup(maneuver)
    init = draw_start_state(maneuver)
    domain = build_model()
    set_init!(domain)
    cas = build_cas(domain)
    solver = solve(cas)
end
