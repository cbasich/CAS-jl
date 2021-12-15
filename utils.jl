using CSV
using DataFrames
using GLM
using Lathe
using MLBase

function record_data(data, filepath)
    CSV.write(filepath, data, delim = ';', append=true)
end

data = [[1; 1; 0; '←'; 1]]

record_data(data, "av_navigation\\data\\test.csv")
