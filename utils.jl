using CSV
using DataFrames
using FeatureSelectors
# using Flux
# using GLM

softmax(x; dims=1) = exp.(x) ./ sum(exp.(x), dims=dims)

function record_data(data, filepath, append=true)
    CSV.write(filepath, DataFrame(data), false)
end
function record_data(data::DataFrame, filepath, append=false)
    CSV.write(filepath, data, append=append)
end
function record_data!(data, df)
    df = push!(copy(df), data)
    # catch
    #     # println(df)
    #     println(data)
    # end
    return df
end

function init_pass_obstacle_data(filepath)
    data = DataFrame(position=Int[], oncoming=Int[], priority=Bool[], σ=Bool[])
    # push!(data, [4 0 true true])
    for pos in 0:4
        for o in -1:3
            if pos < 1 && o != -1
                continue
            end
            for p in 0:1
                for y in 0:1
                    push!(data, [pos o p y])
                    # data = vcat(data, [pos o p y])
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

function update_data!(C, action)
    # D, D_full = C.𝒮.F.D[string(action.value)], C.𝒮.F.D_full[string(action.value)]
    C.𝒮.F.D[string(action.value)] = C.𝒮.F.D_full[string(action.value)][!, vec(hcat(C.𝒮.D.F_active, :σ))]
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

    denom = sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))
    if denom == 0.
        denom = 1
    end
    mcc = (TP * TN - FP * FN) / denom
    return mcc
end
