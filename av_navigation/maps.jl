function generate_dummy_graph()
    nodes = Dict(1 => Dict("pedestrian probability" => .5,
                           "vehicle probabilities" => [.4, .3, .2, .1],
                           "occlusion probability" => 0.5,
                           '→' => 2,
                           '←' => -1,
                           '↑' => -1,
                           '↓' => -1),
                 2 => Dict("pedestrian probability" => .5,
                            "vehicle probabilities" => [.4, .3, .2, .1],
                            "occlusion probability" => 0.5,
                            '→' => -1,
                            '←' => 1,
                            '↑' => -1,
                            '↓' => 3),
                 3 => Dict("pedestrian probability" => .5,
                            "vehicle probabilities" => [.4, .3, .2, .1],
                            "occlusion probability" => 0.5,
                            '→' => -1,
                            '←' => -1,
                            '↑' => 2,
                            '↓' => -1))
    edges = Dict(1 => Dict(2 => Dict("length" => 3,
                                     "direction" => '→',
                                     "num lanes" => 1,
                                     "obstruction probability" => 0.5)),
                 2 => Dict(1 => Dict("length" => 3,
                                     "direction" => '←',
                                     "num lanes" => 1,
                                     "obstruction probability" => 0.5),
                           3 => Dict("length" => 5,
                                     "direction" => '↓',
                                     "num lanes" => 2,
                                     "obstruction probability" => 0.7)),
                 3 => Dict(2 => Dict("length" => 5,
                                     "direction" => '↑',
                                     "num lanes" => 2,
                                     "obstruction probability" => 0.1)))
    G = Graph(nodes, edges)
end

function generate_ma_graph()
    nodes = Dict{Int, Dict{Any, Any}}(
                 1 => Dict("pedestrian probability" => .05,
                           "vehicle probabilities" => [.05, .05, .1, .8],
                           "occlusion probability" => 0.1),
                 2 => Dict("pedestrian probability" => .1,
                           "vehicle probabilities" => [.1, .3, .4, .2],
                           "occlusion probability" => 0.1),
                 3 => Dict("pedestrian probability" => .2,
                           "vehicle probabilities" => [.05, .05, .2, .7],
                           "occlusion probability" => 0.8),
                 4 => Dict("pedestrian probability" => .1,
                           "vehicle probabilities" => [.3, .4, .1, .2],
                           "occlusion probability" => 0.1),
                 5 => Dict("pedestrian probability" => .05,
                           "vehicle probabilities" => [.2, .2, .3, .3],
                           "occlusion probability" => 0.3),
                 6 => Dict("pedestrian probability" => .05,
                           "vehicle probabilities" => [.4, .3, .2, .1],
                           "occlusion probability" => 0.1),
                 7 => Dict("pedestrian probability" => .7,
                           "vehicle probabilities" => [.3, .05, .05, .6],
                           "occlusion probability" => 0.5),
                 8 => Dict("pedestrian probability" => .1,
                           "vehicle probabilities" => [.3, .2, .4, .1],
                           "occlusion probability" => 0.1),
                 9 => Dict("pedestrian probability" => .5,
                           "vehicle probabilities" => [.3, .1, .2, .4],
                           "occlusion probability" => 0.3),
                10 => Dict("pedestrian probability" => .9,
                           "vehicle probabilities" => [.3, .1, .2, .4],
                           "occlusion probability" => 0.1),
                11 => Dict("pedestrian probability" => .9,
                           "vehicle probabilities" => [.3, .1, .2, .4],
                           "occlusion probability" => 0.1),
                12 => Dict("pedestrian probability" => .9,
                           "vehicle probabilities" => [.1, .1, .3, .5],
                           "occlusion probability" => 0.1),
                13 => Dict("pedestrian probability" => .05,
                           "vehicle probabilities" => [.3, .05, .05, .6],
                           "occlusion probability" => 0.1),
                14 => Dict("pedestrian probability" => .1,
                           "vehicle probabilities" => [.3, .05, .05, .6],
                           "occlusion probability" => 0.1),
                15 => Dict("pedestrian probability" => .1,
                           "vehicle probabilities" => [.3, .1, .2, .4],
                           "occlusion probability" => 0.1),
                16 => Dict("pedestrian probability" => .5,
                           "vehicle probabilities" => [.1, .1, .1, .7],
                           "occlusion probability" => 0.4))

    edges = Dict{Int, Dict{Int, Dict{String, Any}}}(
                 1 => Dict( 2 => Dict("length" => 21,
                                      "direction" => '→',
                                      "num lanes" => 2,
                                      "obstruction probability" => 0.1),
                            13 => Dict("length" => 102,
                                      "direction" => '↓',
                                      "num lanes" => 3,
                                      "obstruction probability" => 0.05)),
                  2 => Dict(1 => Dict("length" => 21,
                                      "direction" => '←',
                                      "num lanes" => 2,
                                      "obstruction probability" => 0.1),
                            3 => Dict("length" => 21,
                                      "direction" => '→',
                                      "num lanes" => 2,
                                      "obstruction probability" => 0.3),
                            5 => Dict("length" => 94,
                                      "direction" => '↓',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.5)),
                  3 => Dict(2 => Dict("length" => 21,
                                      "direction" => '←',
                                      "num lanes" => 2,
                                      "obstruction probability" => 0.3),
                            4 => Dict("length" => 14,
                                      "direction" => '→',
                                      "num lanes" => 2,
                                      "obstruction probability" => 0.3),
                            6 => Dict("length" => 38,
                                      "direction" => '↓',
                                      "num lanes" => 2,
                                      "obstruction probability" => 0.2)),
                    4=>Dict(3 => Dict("length" => 14,
                                      "direction" => '←',
                                      "num lanes" => 2,
                                      "obstruction probability" => 0.3),
                            7 => Dict("length" => 41,
                                      "direction" => '↓',
                                      "num lanes" => 2,
                                      "obstruction probability" => 0.2)),
                    5=>Dict(2 => Dict("length" => 94,
                                      "direction" => '↑',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.5),
                            6 => Dict("length" => 22,
                                      "direction" => '→',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.5),
                            9 => Dict("length" => 8,
                                      "direction" => '↓',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.1)),
                    6=>Dict(3 => Dict("length" => 38,
                                      "direction" => '↑',
                                      "num lanes" => 3,
                                      "obstruction probability" => 0.2),
                            5 => Dict("length" => 22,
                                      "direction" => '←',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.5),
                            7 => Dict("length" => 14,
                                      "direction" => '→',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.1),
                            15=> Dict("length" => 16,
                                      "direction" => '↓',
                                      "num lanes" => 3,
                                      "obstruction probability" => 0.1)),
                    7=>Dict(4 => Dict("length" => 41,
                                      "direction" => '↑',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.1),
                            6 => Dict("length" => 14,
                                      "direction" => '←',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.1),
                            8 => Dict("length" => 13,
                                      "direction" => '→',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.35)),
                    8=>Dict(7 => Dict("length" => 13,
                                      "direction" => '←',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.35),
                            10=> Dict("length" => 4,
                                      "direction" => '↓',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.1)),
                    9=>Dict(5 => Dict("length" => 8,
                                      "direction" => '↑',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.1),
                            14=> Dict("length" => 12,
                                      "direction" => '↓',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.2)),
                    10=>Dict(8=> Dict("length" => 4,
                                      "direction" => '↑',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.1),
                            11=> Dict("length" => 5,
                                      "direction" => '↓',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.3)),
                    11=>Dict(10=>Dict("length" => 5,
                                      "direction" => '↑',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.3),
                            16=> Dict("length" => 2,
                                      "direction" => '↓',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.2)),
                    12=>Dict(13=>Dict("length" => 11,
                                      "direction" => '→',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.5)),
                    13=>Dict(1 =>Dict("length" => 102,
                                      "direction" => '↑',
                                      "num lanes" => 3,
                                      "obstruction probability" => 0.1),
                            12=> Dict("length" => 11,
                                      "direction" => '←',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.5),
                            14=> Dict("length" => 15,
                                      "direction" => '→',
                                      "num lanes" => 2,
                                      "obstruction probability" => 0.2)),
                    14=>Dict(9=> Dict("length" => 11,
                                      "direction" => '↑',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.2),
                            13=> Dict("length" => 15,
                                      "direction" => '←',
                                      "num lanes" => 2,
                                      "obstruction probability" => 0.2),
                            15=> Dict("length" => 33,
                                      "direction" => '→',
                                      "num lanes" => 2,
                                      "obstruction probability" => 0.3)),
                    15=>Dict(6=> Dict("length" => 16,
                                      "direction" => '↑',
                                      "num lanes" => 3,
                                      "obstruction probability" => 0.1),
                            14=> Dict("length" => 33,
                                      "direction" => '←',
                                      "num lanes" => 2,
                                      "obstruction probability" => 0.3),
                            16=> Dict("length" => 16,
                                      "direction" => '→',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.2)),
                    16=>Dict(11=>Dict("length" => 2,
                                      "direction" => '↑',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.2),
                            15=> Dict("length" => 16,
                                      "direction" => '←',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.2)))

    for n in keys(nodes)
        for dir in ['↑', '→', '↓', '←']
            nodes[n][dir] = -1

            E = edges[n]
            for (e, v) in E
                θ = v["direction"]
                nodes[n][θ] = e
            end
        end
    end
    return Graph(nodes, edges)
end

fixed_routes = Dict(
    "Northampton to Amherst 1" => (12, 10),
    "Amherst 1 to Northampton" => (10, 12),
    "Northampton to Amherst 2" => (12, 11),
    "Amherst 2 to Northampton" => (11, 12),
    "Northampton to UMass" => (12, 7),
    "UMass to Northampton" => (7, 12),
    "Amherst 1 to UMass" => (10, 7),
    "UMass to Amherst 1" => (7, 10),
    "Amherst 2 to UMass" => (11, 7),
    "UMass to Amherst 2" => (7, 11),
    "Sunderland to UMass" => (2, 7),
    "UMass to Sunderland" => (7, 2),
    "Amherst 1 to Sunderland" => (11, 2),
    "Sunderland to Amherst 1" => (2, 11)
)
