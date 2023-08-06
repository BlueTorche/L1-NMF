# import Pkg;
# Pkg.add("SparseArrays")
# Pkg.add("MAT")
# Pkg.add("DelimitedFiles")
# Pkg.add("Plots")
# Pkg.add("Munkres")

using .L1NMF

using MAT
using SparseArrays
using DelimitedFiles
using Munkres
using Plots


function preprossessing(X)
    m,_ = size(X)
    for p in 1:m
        somme = length(findall(x -> x > 0, X[p,:]))
        X[p,:] *= log(m/somme)
    end
    return X
end

# Get Data
data  = matopen("./k1b.mat")
classid = read(data, "classid")
X = Matrix{Float64}(sparse(read(data,"dtm")))'
X = preprossessing(X)
r = convert(Int64, maximum(classid))
close(data)

lambda=1.0
benchmark = true


W, H, times1,errors1 = L1NMF.l1_sparse_nmf(X,r,lambda = lambda, benchmark=benchmark)

