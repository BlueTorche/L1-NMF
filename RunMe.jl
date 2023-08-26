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

function Accuracy(TrueClass,H)
    r = convert(Int64, maximum(TrueClass))
    Mat = zeros(r,r)
    for k in eachindex(TrueClass)
        max_index = argmax(H[:,k])
        maxi      = H[max_index,k]
        i = convert(Int64,TrueClass[k])
        for j in 1:r
            if maxi == 0 || max_index != j
                Mat[i,j]+=1.0
            end
        end
    end

    best_jobs = munkres(Mat)

    acc = 0
    for i in 1:r
        acc += Mat[i,best_jobs[i]]
    end
    return 1 - acc/length(TrueClass), best_jobs
end

# Get Data
data  = matopen("datasets/reviews.mat")
classid = read(data, "classid")
X = Matrix{Float64}(sparse(read(data,"dtm")))'
close(data)

X = preprossessing(X)
r = convert(Int64, maximum(classid))
m, n      = size(X)

benchmark = true

# L2NMF model
W, H, times1,errors1 = L1NMF.l2nmf(X, r, benchmark=benchmark, maxiter=20)

Acc,_ = Accuracy(classid, H)
println("Accuracy with l2nmf : $Acc")

# Global Lambda L1NMF model
lambda=1.0

W, H, times1,errors1 = L1NMF.global_lambda_sparse_l1nmf(X, r, lambda = lambda, benchmark=benchmark)

Acc,_ = Accuracy(classid, H)
println("Accuracy with l1nmf and global lambda : $Acc")


# Local Lambda L1NMF mdoel
alpha = 0.25

lambdaH = zeros(n)
K = length(findall(x -> x > 0, X))
for i in 1:n
    Kj = findall(x -> x > 0, X[:,i])
    lambdaH[i] = n*length(Kj)/K*alpha
end

W, H, timesCol,errorsCol = L1NMF.local_lambda_sparse_l1nmf(X, r, lambdaH = lambdaH, benchmark=benchmark)
Acc,_ = Accuracy(classid, H)
println("Accuracy with l1nmf and local lambda : $Acc")
