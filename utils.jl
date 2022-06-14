using CSV
using DataFrames
# using Flux
# using GLM

softmax(x; dims=1) = exp.(x) ./ sum(exp.(x), dims=dims)

function record_data(data, filepath, append=true)
    CSV.write(filepath, DataFrame(data, :auto), append=append, header=false)
end
function record_data(data::DataFrame, filepath, append=false)
    CSV.write(filepath, data, append=append, header=false)
end
function record_data!(data, df)
    try
        push!(df, data)
    catch
        println(df)
        println(data)
    end
end

function init_pass_obstacle_data(filepath)
    data = [:x1 :x2 :x3 :x4 :y]
    for pos in 0:4
        for o in -1:3
            if pos < 1 && o != -1
                continue
            end
            for t in 0:1
                for p in 0:1
                    for y in 0:1
                        data = vcat(data, [pos o t p y])
                    end
                end
            end
        end
    end
    record_data(data, filepath, false)
end

function init_node_data(filepath)
    d = [:x1 :x2 :x3 :x4 :y; 1 1 1 1 1]
    # for p in 0:1
    #     for o in 0:1
    #         for v in 0:4
    #             for l in 1:2
    #                 for y in 0:1
    #                     d = vcat(d, [p o v l y])
    #                 end
    #             end
    #         end
    #     end
    # end
    record_data(d,filepath,false)
end

function init_edge_data(filepath)
    d = [:x1 :x2 :x3 :y; 1 1 1 1]
    # for o in 0:1
    #     for n in 1:3
    #         for l in 1:2
    #             for y in 0:1
    #                 d = vcat(d, [o n l y])
    #             end
    #         end
    #     end
    # end
    record_data(d, filepath, false)
end

function read_data(filepath)
    df = DataFrame(CSV.File(filepath))
    X = select(df, Not(last(names(df))))
    rename!(X, [Symbol("x$i") for i in 1:size(names(X))[1]])
    Y = select(df, last(names(df)))
    rename!(Y, [:y])
    return X, Y
end

function split_data(df::DataFrame)
    X = select(df, Not(last(names(df))))
    rename!(X, [Symbol("x$i") for i in 1:size(names(X))[1]])
    Y = select(df, last(names(df)))
    rename!(Y, [:y])
    return X, Y
end

# D = DataFrame(CSV.File("pass_obstacle\\data\\go.csv"))


# function OneHot(data)
#     if size(unique(data[:, 1])) == (2,)
#         OHE = DataFrame(x=data[:, 1])
#     else
#         df = DataFrame(x=data[:, 1])
#         OHE = select(df, [:x => ByRow(isequal(v))=>Symbol(v) for v in unique(df.x)])
#     end
#
#     for i = 2:size(data)[2]
#         if size(unique(data[:, i])) == (2,)
#             OHE = hcat(OHE, DataFrame(:x$i = data[:, i]))
#         else
#             df = DataFrame(x=data[:, i])
#             tmp = select(df, [:x => ByRow(isequal(v))=>Symbol(v) for v in sort(unique(df.x))])
#             OHE = hcat(OHE, tmp)
#         end
#     end
#
#     return OHE
#     return DataFrame(OHE)
# end
