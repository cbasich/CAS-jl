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
    # println(df)
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
    data = DataFrame(p=Bool[], o=Bool[], v=Int[], level=Int[], σ=Bool[])
    for p in 0:1
        for o in 0:1
            for v in 0:4
                for l in 1:2
                    for y in 0:1
                        push!(data, [p o v l y])
                        # d = vcat(d, [p o v l y])
                    end
                end
            end
        end
    end
    record_data(data,filepath,false)
end

function init_edge_data(filepath)
    data = DataFrame(o=Bool[], l=Int[], level=Int[], σ=Bool[])
    for o in 0:1
        for n in 1:3
            for l in 1:2
                for y in 0:1
                    push!(data, [o n l y])
                    # d = vcat(d, [o n l y])
                end
            end
        end
    end
    record_data(data, filepath, false)
end

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

function mRMR(df, y)
    relevance = 0.
    if length(unique(df)) == 1
        return 0.0
    elseif length(unique(df)) == 2
        X = convert(Array{Bool, 1}, df .== unique(df)[1])
        X = reshape(X, :, 1)
        return pearson_correlation(X, y)[1]
    elseif typeof(df) == Array{Bool, 1}
        X = reshape(df, :, 1)
    else
        X = Matrix(one_hot_encode(DataFrame([df]), drop_original=true))
    end
    relevance = sum(pearson_correlation(X, y)) / size(X)[2] #f_test

    repetition = 0.
    for i = 1:size(X)[2]
        repetition += sum(pearson_correlation(X, X[:, i])) - 1.0
    end
    repetition = repetition / size(X)[2]
    return relevance / repetition
end

function build_lambda(D_train, features, discriminator)
    println("Features: ", features)
    println("Discriminator: ", discriminator)
    X = Matrix(D_train[!, vec(hcat(features, discriminator))])
    # catch
    #     println("Features: ", features)
    #     println("Discriminator: ", discriminator)
    # end
    Y = D_train[:, :σ]
    λ = build_forest(Y, X, -1, 11, 0.7, -1)
    return λ
end

function test_lambda(λ, D, features, discriminator)
    if discriminator == -1
        X = Matrix(D[!, vec(features)])
    else
        X = Matrix(D[!, vec(hcat(features, discriminator))])
    end
    Y = D[!, :σ]
    preds = apply_forest_proba(λ, X, [0,1])[:, 2] .> 0.5
    return mcc(Y, preds)
end

function test_discriminators(C, D, D_full, D_train, D_test, F, discriminators)
    lambdas = [build_lambda(D_train, F, d[1]) for d in discriminators]
    scores = [test_lambda(lambdas[i], D_test, F, discriminators[i][1]) for i=1:length(lambdas)]

    # scores = []
    # for d in discriminators
    #     # @show vec(hcat(F, d[1]))
    #     X = Matrix(D_full[!, vec(hcat(F, d[1]))])
    #     Y = D_test[!, :σ]
    #     r2 = nfoldCV_forest(Y, X, 3, -1, 10, 0.7, -1; verbose=false)
    #     push!(scores, mean(r2))
    # end

    best = argmax(scores)
    best_score = scores[best]
    best_discriminator = discriminators[best][1]

    # X₀, Y₀ = split_data(D)
    X₀, Y₀ = D_train[!, vec(F)], D_train[:, :σ]
    λ₀ = build_forest(Y₀, Matrix(X₀), -1, 11, 0.7, -1)
    curr_score = test_lambda(λ₀, D_test, F, -1)

    # X₀, Y₀ = Matrix(D_full[!, vec(F)]), D_full[!, :σ]
    # curr_score = mean(nfoldCV_forest(Y₀, X₀, 3, -1, 10, 0.7, -1; verbose=false))
    println("Curr score: $curr_score")
    println("Best score: $best_score")
    if best_score > curr_score + 0.2
        return best_discriminator
    else
        return -1
    end
    # if best_score < curr_score + 0.1 || best_score < 0.5 || curr_score == -1.0
    #     return -1
    # else
    #     return best_discriminator
    # end
end

function update_world_state!(wstate, state, action, state′)
    wstate′ = deepcopy(wstate)
    if typeof(state′) == NodeState
        if wstate′.lanes == 2
            wstate.left_occupied = sample([false, true], aweights([0.5, 0.5]))
            if wstate′.left_occupied == true
                wstate.right_occupied = false
            else
                wstate.right_occupied = sample([false, true], aweights([0.5, 0.5]))
            end
        elseif wstate′.lanes == 3
            wstate.left_occupied = sample([false, true], aweights([0.5, 0.5]))
            wstate.right_occupied = sample([false, true], aweights([0.5, 0.5]))
        end
        if typeof(state) == EdgeState
            wstate.trailing = sample([false, true], aweights([0.5, 0.5]))
            wstate.waiting = false
        else
            if wstate′.trailing == false
                wstate.trailing = sample([false, true], aweights([0.5, 0.5]))
            end
            wstate.waiting = true
        end
    else
        wstate.lanes = state′.l
        wstate.trailing = sample([false, true], aweights([0.5, 0.5]))
        wstate.left_occupied = sample([false, true], aweights([0.5, 0.5]))
        wstate.right_occupied = sample([false, true], aweights([0.5, 0.5]))
        if typeof(state) == NodeState
            wstate.waiting = false
        else
            wstate.waiting = (state.o == state′.o == true)
        end
    end
    # wstate = wstate′
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
