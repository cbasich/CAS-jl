using CSV
using DataFrames
# using Flux
# using GLM

softmax(x; dims=1) = exp.(x) ./ sum(exp.(x), dims=dims)

function record_data(data, filepath, append=true)
    CSV.write(filepath, DataFrame(data, :auto), append=append, header=false)
end
function record_data(data::DataFrame, filepath, append=false)
    CSV.write(filepath, data, append=append, header=true)
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
    data = [:position :oncoming :priority :σ]
    for pos in 0:4
        for o in -1:3
            if pos < 1 && o != -1
                continue
            end
            for p in 0:1
                for y in 0:1
                    data = vcat(data, [pos o p y])
                end
            end
        end
    end
    record_data(data, filepath, false)
end

function init_node_data(filepath)
    d = [:x1 :x2 :x3 :x4 :y]
    for p in 0:1
        for o in 0:1
            for v in 0:4
                for l in 1:2
                    for y in 0:1
                        d = vcat(d, [p o v l y])
                    end
                end
            end
        end
    end
    record_data(d,filepath,false)
end

function init_edge_data(filepath)
    d = [:x1 :x2 :x3 :y]
    for o in 0:1
        for n in 1:3
            for l in 1:2
                for y in 0:1
                    d = vcat(d, [o n l y])
                end
            end
        end
    end
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
    return Matrix(X), Array(Y[!, :y])
end

function split_df(df::DataFrame, pct)
    @assert 0 <= pct <= 1
    ids = collect(axes(df, 1))
    shuffle!(ids)
    sel = ids .<= nrow(df) .* pct
    return view(df, sel, :), view(df, .!sel, :)
end

function mcc(y1, y2)
    TP = sum(y1 .== y2 .== trues(length(y1)))
    TN = sum(y1 .== y2 .== falses(length(y1)))
    FP = sum(y2) - TP
    FN = length(y2) - (TP + TN + FP)

    mcc = (TP * TN - FP * FN) / sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))
    return mcc
end
