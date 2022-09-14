using CSV
using DataFrames
using JLD2
using StatsBase
using CSV
using DecisionTree
using DataFrames

softmax(x; dims=1) = exp.(x) ./ sum(exp.(x), dims=dims)

function record_data(data, filepath, append=true)
    CSV.write(filepath, DataFrame(data), false)
end
function record_data(data::DataFrame, filepath, append=false)
    CSV.write(filepath, data, append=append)
end
function record_data!(data, df)
    # println(df)
    df = push!(copy(df), data)
    # catch
    #     # println(df)
    #     println(data)
    # end
    return df
end

function init_node_data(filepath)
    CSV.write(filepath,
        DataFrame([false false 0 1 "day" "sunny" 1 1 1]);
        header = vec(hcat(:p, :o, :v, :a, :t, :w, :sh, :level, :σ)))
    # data = DataFrame(p=Bool[], o=Bool[], v=Int[], a=Int[], t=String[], w=String[], sh=Int[], level=Int[], σ=Bool[])
    # for p in 0:1
    #     for o in 0:1
    #         for v in 0:3
    #             for a in 1:4
    #                 for t in ["day", "night"]
    #                     for w in ["sunny", "rainy", "snowy"]
    #                         for sh=1:2
    #                             for l in 0:1
    #                                 for y in 0:1
    #                                     push!(data, [p o v a t w sh l y])
    #                                     # d = vcat(d, [p o v l y])
    #                                 end
    #                             end
    #                         end
    #                     end
    #                 end
    #             end
    #         end
    #     end
    # end
    # record_data(data,filepath,false)
end

function init_cas_node_data(filepath)
    CSV.write(filepath,
        DataFrame([false false 0 1 "day" "sunny" 1 1]);
        header = vec(hcat(:p, :o, :v, :a, :t, :w, :level, :σ)))
end

function init_edge_data(filepath)
    CSV.write(filepath,
        DataFrame([false 1 1 "day" "sunny" 1 1 1]);
        header = vec(hcat(:o, :l, :a, :t, :w, :sh, :level, :σ)))
    # data = DataFrame(o=Bool[], l=Int[], a=Int[], t=String[], w=String[], sh=Int[], level=Int[], σ=Bool[])
    # for o in 0:1
    #     for n in 1:3
    #         for a=1:4
    #             for t in ["day", "night"]
    #                 for w in ["sunny", "rainy", "snowy"]
    #                     for sh=0:1
    #                         for l in 0:1
    #                             for y in 0:1
    #                                 push!(data, [o n a t w sh l y])
    #                                 # d = vcat(d, [o n l y])
    #                             end
    #                         end
    #                     end
    #                 end
    #             end
    #         end
    #     end
    # end
    # record_data(data, filepath, false)
end

function init_cas_edge_data(filepath)
    CSV.write(filepath,
        DataFrame([false 1 1 "day" "sunny" 1 1]);
        header = vec(hcat(:o, :l, :a, :t, :w, :level, :σ)))
end

# function init_edge_data(filepath)
#     data = DataFrame(o=Bool[], l=Int[], level=Int[], σ=Bool[])
#     for o in 0:1
#         for n in 1:3
#             for l in 1:2
#                 for y in 0:1
#                     push!(data, [o n l y])
#                     # d = vcat(d, [o n l y])
#                 end
#             end
#         end
#     end
#     record_data(data, filepath, false)
# end

function update_data!(C, action)
    # D, D_full = C.𝒮.F.D[string(action.value)], C.𝒮.F.D_full[string(action.value)]
    C.𝒮.F.D[string(action.value)] = C.𝒮.F.D_full[string(action.value)][!, vec(hcat(C.𝒮.D.F_active, :σ))]
end

function update_data!(C, action, statetype)
    F = C.𝒮.D.F_active[statetype]
    C.𝒮.F.D[statetype][string(action.value)] = C.𝒮.F.D_full[statetype][string(action.value)][!, vec(hcat(F, :level, :σ))]
end

function read_data(filepath)
    df = DataFrame(CSV.File(filepath))
    X = select(df, Not(last(names(df))))
    rename!(X, [Symbol("x$i") for i in 1:size(names(X))[1]])
    Y = select(df, last(names(df)))
    rename!(Y, [:y])
    return X, Y
end

function split_data(df::Union{DataFrame, SubDataFrame})
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

function smooth_data(x,w)
    smoothed = []
    for i=6:length(x)
        trunc_data = x[1:i]
        if i < w+1
            append!(smoothed, mean(x[1:i]))
        else
            window = trunc_data[end-w:end]
            append!(smoothed, mean(window))
        end
    end
    return smoothed
end
