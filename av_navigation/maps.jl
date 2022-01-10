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
    nodes = Dict{Int, Dict{Any, Any}}(1 => Dict("pedestrian probability" => .05,
                           "vehicle probabilities" => [.3, .1, .2, .4],
                           "occlusion probability" => 0.5),
                 2 => Dict("pedestrian probability" => .05,
                           "vehicle probabilities" => [.3, .1, .2, .4],
                           "occlusion probability" => 0.5),
                 3 => Dict("pedestrian probability" => .05,
                           "vehicle probabilities" => [.3, .1, .2, .4],
                           "occlusion probability" => 0.5),
                 4 => Dict("pedestrian probability" => .05,
                           "vehicle probabilities" => [.3, .1, .2, .4],
                           "occlusion probability" => 0.5),
                 5 => Dict("pedestrian probability" => .05,
                           "vehicle probabilities" => [.3, .1, .2, .4],
                           "occlusion probability" => 0.5),
                 6 => Dict("pedestrian probability" => .05,
                           "vehicle probabilities" => [.3, .1, .2, .4],
                           "occlusion probability" => 0.5),
                 7 => Dict("pedestrian probability" => .05,
                           "vehicle probabilities" => [.3, .1, .2, .4],
                           "occlusion probability" => 0.5),
                 8 => Dict("pedestrian probability" => .05,
                           "vehicle probabilities" => [.3, .1, .2, .4],
                           "occlusion probability" => 0.5),
                 9 => Dict("pedestrian probability" => .05,
                           "vehicle probabilities" => [.3, .1, .2, .4],
                           "occlusion probability" => 0.5),
                10 => Dict("pedestrian probability" => .05,
                           "vehicle probabilities" => [.3, .1, .2, .4],
                           "occlusion probability" => 0.5),
                11 => Dict("pedestrian probability" => .05,
                           "vehicle probabilities" => [.3, .1, .2, .4],
                           "occlusion probability" => 0.5),
                12 => Dict("pedestrian probability" => .05,
                           "vehicle probabilities" => [.3, .1, .2, .4],
                           "occlusion probability" => 0.5),
                13 => Dict("pedestrian probability" => .05,
                           "vehicle probabilities" => [.3, .1, .2, .4],
                           "occlusion probability" => 0.5),
                14 => Dict("pedestrian probability" => .05,
                           "vehicle probabilities" => [.3, .1, .2, .4],
                           "occlusion probability" => 0.5),
                15 => Dict("pedestrian probability" => .05,
                           "vehicle probabilities" => [.3, .1, .2, .4],
                           "occlusion probability" => 0.5),
                16 => Dict("pedestrian probability" => .05,
                           "vehicle probabilities" => [.3, .1, .2, .4],
                           "occlusion probability" => 0.5))

    edges = Dict(1 => Dict(2 => Dict("length" => 21,
                                      "direction" => '→',
                                      "num lanes" => 2,
                                      "obstruction probability" => 0.1),
                            13 => Dict("length" => 102,
                                      "direction" => '↓',
                                      "num lanes" => 3,
                                      "obstruction probability" => 0.1)),
                  2 => Dict(1 => Dict("length" => 21,
                                      "direction" => '←',
                                      "num lanes" => 2,
                                      "obstruction probability" => 0.1),
                            3 => Dict("length" => 21,
                                      "direction" => '→',
                                      "num lanes" => 2,
                                      "obstruction probability" => 0.1),
                            5 => Dict("length" => 94,
                                      "direction" => '↓',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.1)),
                  3 => Dict(2 => Dict("length" => 21,
                                      "direction" => '←',
                                      "num lanes" => 2,
                                      "obstruction probability" => 0.1),
                            4 => Dict("length" => 14,
                                      "direction" => '→',
                                      "num lanes" => 2,
                                      "obstruction probability" => 0.1),
                            6 => Dict("length" => 38,
                                      "direction" => '↓',
                                      "num lanes" => 3,
                                      "obstruction probability" => 0.1)),
                    4=>Dict(3 => Dict("length" => 14,
                                      "direction" => '←',
                                      "num lanes" => 2,
                                      "obstruction probability" => 0.1),
                            7 => Dict("length" => 41,
                                      "direction" => '↓',
                                      "num lanes" => 2,
                                      "obstruction probability" => 0.1)),
                    5=>Dict(2 => Dict("length" => 94,
                                      "direction" => '↑',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.1),
                            6 => Dict("length" => 22,
                                      "direction" => '→',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.1),
                            9 => Dict("length" => 8,
                                      "direction" => '↓',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.1)),
                    6=>Dict(3 => Dict("length" => 38,
                                      "direction" => '↑',
                                      "num lanes" => 3,
                                      "obstruction probability" => 0.1),
                            5 => Dict("length" => 22,
                                      "direction" => '←',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.1),
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
                                      "obstruction probability" => 0.1)),
                    8=>Dict(7 => Dict("length" => 13,
                                      "direction" => '←',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.1),
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
                                      "obstruction probability" => 0.1)),
                    10=>Dict(8=> Dict("length" => 4,
                                      "direction" => '↑',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.1),
                            11=> Dict("length" => 5,
                                      "direction" => '↓',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.1)),
                    11=>Dict(10=>Dict("length" => 5,
                                      "direction" => '↑',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.1),
                            16=> Dict("length" => 2,
                                      "direction" => '↓',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.1)),
                    12=>Dict(13=>Dict("length" => 11,
                                      "direction" => '→',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.1)),
                    13=>Dict(1 =>Dict("length" => 102,
                                      "direction" => '↑',
                                      "num lanes" => 3,
                                      "obstruction probability" => 0.1),
                            12=> Dict("length" => 11,
                                      "direction" => '←',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.1),
                            14=> Dict("length" => 15,
                                      "direction" => '→',
                                      "num lanes" => 2,
                                      "obstruction probability" => 0.1)),
                    14=>Dict(9=> Dict("length" => 11,
                                      "direction" => '↑',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.1),
                            13=> Dict("length" => 15,
                                      "direction" => '←',
                                      "num lanes" => 2,
                                      "obstruction probability" => 0.1),
                            15=> Dict("length" => 33,
                                      "direction" => '→',
                                      "num lanes" => 2,
                                      "obstruction probability" => 0.1)),
                    15=>Dict(6=> Dict("length" => 16,
                                      "direction" => '↑',
                                      "num lanes" => 3,
                                      "obstruction probability" => 0.1),
                            14=> Dict("length" => 33,
                                      "direction" => '←',
                                      "num lanes" => 2,
                                      "obstruction probability" => 0.1),
                            16=> Dict("length" => 16,
                                      "direction" => '→',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.1)),
                    16=>Dict(11=>Dict("length" => 2,
                                      "direction" => '↑',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.1),
                            15=> Dict("length" => 16,
                                      "direction" => '←',
                                      "num lanes" => 1,
                                      "obstruction probability" => 0.1)))

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
