using CSV
using DataFrames
using Flux
using GLM

function init_navigation_data(action)
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
    record_data(d, "av_navigation\\data\\$action.csv")
end

function init_navigation_data(action)
    d = [:x1 :x2 :x3 :y]
    for o in 0:1
        for n in 1:2
            for l in 1:2
                for y in 0:1
                    d = vcat(d, [o n l y])
                end
            end
        end
    end
    record_data(d, "av_navigation\\data\\$action.csv")
end

function record_data(data, filepath)
    CSV.write(filepath, DataFrame(data, :auto), append=true, header=false)
end

function read_data(filepath)
    df = DataFrame(CSV.File(filepath))
    X = select(df, Not(last(names(df))))
    rename!(X, [Symbol("x$i") for i in 1:size(names(X))[1]])
    Y = select(df, last(names(df)))
    rename!(Y, [:y])
    return X, Y
end

X, Y = read_data("av_navigation\\data\\↑.csv")
@show X
@show Y
fm = @formula(y ~ x1 + x2 + x3 + x4 + x5)
fm = @formula(y ~ col1 + col2 + col3 + col4 + col5)
logit = glm(fm, hcat(X, Y), Binomial(), ProbitLink())

data = hcat(rand(0:1, 20), rand(0:1, 20), rand(0:4, 20), rand(["←", "↑", "→", "↓"], 20), rand(1:2, 20))
@show data
@show d = DataFrame(data)
rename!(d, [Symbol("x$i") for i in 1:size(names(d))[1]])
@show d
d[!, :col1] = convert.(Int64, d[:, :col1])
d[!, :col2] = convert.(Int64, d[:, :col2])
d[!, :col3] = convert.(Int64, d[:, :col3])
d[!, :col4] = convert.(String3, d[:, :col4])
d[!, :col5] = convert.(Int64, d[:, :col5])
@show X[1:20, :]

@show predict(logit, d)
@show predict(logit, DataFrame([1 0 2 "→" 1]))[1]

function OneHot(data)
    if size(unique(data[:, 1])) == (2,)
        OHE = DataFrame(label1=data[:, 1])
    else
        df = DataFrame(x=data[:, 1])
        OHE = select(df, [:x => ByRow(isequal(v))=>Symbol(v) for v in unique(df.x)])
    end

    for i = 2:size(data)[2]
        if size(unique(data[:, i])) == (2,)
            OHE = hcat(OHE, DataFrame(label2=data[:, i]))
        else
            df = DataFrame(x=data[:, i])
            tmp = select(df, [:x => ByRow(isequal(v))=>Symbol(v) for v in sort(unique(df.x))])
            OHE = hcat(OHE, tmp)
        end
    end

    return OHE
    return DataFrame(OHE)
end
