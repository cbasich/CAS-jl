using CSV
using DataFrames
using Flux

function record_data(data, filepath)
    CSV.write(filepath, DataFrame(data, :auto), append=true, header=false)
end

data = [[true true '←' true];
        [true false '↑' false]]

println(Flux.onehot(data))


record_data(data, "av_navigation\\data\\test.csv")

function OneHot(x::Vector; header::Bool = false, drop::Bool = true)
    UniqueVals = unique(x)  ## note: don't sort this - that will mess up order of vals to idx.
    Val_to_Idx = [Val => Idx for (Idx, Val) in enumerate(unique(x))] ## create a dictionary that maps unique values in the input array to column positions in the new sparse matrix.
    ColIdx = convert(Array{Int64}, [Val_to_Idx[Val] for Val in x])
    MySparse = sparse(collect(1:length(x)),  ColIdx, ones(Int32, length(x)))
    if drop
        StartIdx = 2
    else
        StartIdx = 1
    end
    if header
        return (MySparse[:,StartIdx:end], UniqueVals[StartIdx:end])  ## I.e. gives you back a tuple, second element is the header which you can then feed to something to name the columns or do whatever else with
    else
        return MySparse[:,StartIdx:end]  ## use MySparse[:, 2:end] to drop a value
    end
end
@show data[,1]
# OneHot([1,1,'←',1])

# @show Flux.onehot(DataFrame(data, :auto))

@show unique(DataFrame(data, :auto))
Datafr
